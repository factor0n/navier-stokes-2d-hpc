/*****************************************************************************
 * solver.c - Core Navier-Stokes solver routines
 *
 * Fractional Step Method (Chorin's Projection):
 *   1. Compute intermediate velocity u* (advection + diffusion)
 *   2. Solve pressure Poisson equation: ∇²p = (1/dt) ∇·u*
 *   3. Project: u^{n+1} = u* - dt * ∇p
 *
 * Spatial discretization: 2nd-order central differences
 * Advection: 2nd-order central (stable for moderate Re)
 *****************************************************************************/

#include "ns2d.h"

/* ========================================================================= */
/*  MPI Ghost Row Exchange (1D decomposition in y)                           */
/* ========================================================================= */

void exchange_ghost_rows(double *field, const SimParams *par, int nx_loc)
{
    int rank   = par->rank;
    int nprocs = par->nprocs;
    int count  = nx_loc;

    int rank_below = (rank > 0)          ? rank - 1 : MPI_PROC_NULL;
    int rank_above = (rank < nprocs - 1) ? rank + 1 : MPI_PROC_NULL;

    int ny_loc = par->local_ny + 2;

    MPI_Request reqs[4];
    int nreq = 0;

    /* Send bottom interior row (j=1) to rank below -> their top ghost */
    MPI_Isend(&field[IDX(0, 1, nx_loc)], count, MPI_DOUBLE,
              rank_below, 0, MPI_COMM_WORLD, &reqs[nreq++]);

    /* Receive into bottom ghost row (j=0) from rank below */
    MPI_Irecv(&field[IDX(0, 0, nx_loc)], count, MPI_DOUBLE,
              rank_below, 1, MPI_COMM_WORLD, &reqs[nreq++]);

    /* Send top interior row (j=local_ny) to rank above -> their bottom ghost */
    MPI_Isend(&field[IDX(0, ny_loc - 2, nx_loc)], count, MPI_DOUBLE,
              rank_above, 1, MPI_COMM_WORLD, &reqs[nreq++]);

    /* Receive into top ghost row (j=ny_loc-1) from rank above */
    MPI_Irecv(&field[IDX(0, ny_loc - 1, nx_loc)], count, MPI_DOUBLE,
              rank_above, 0, MPI_COMM_WORLD, &reqs[nreq++]);

    MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);
}

/* ========================================================================= */
/*  Step 1: Compute intermediate velocity u*, v* (Advection + Diffusion)     */
/* ========================================================================= */

void compute_rhs_momentum(FieldData *fld, const SimParams *par)
{
    int nx  = fld->nx_loc;
    int ny  = fld->ny_loc;
    double dx  = par->dx;
    double dy  = par->dy;
    double dt  = par->dt;
    double nu  = par->nu;
    double idx = 1.0 / dx;
    double idy = 1.0 / dy;
    double idx2 = idx * idx;
    double idy2 = idy * idy;

    double *u = fld->u;
    double *v = fld->v;
    double *us = fld->u_star;
    double *vs = fld->v_star;

    /* Exchange ghost rows for u and v before computing */
    exchange_ghost_rows(u, par, nx);
    exchange_ghost_rows(v, par, nx);

    /* u-momentum: u* = u + dt * (-adv_u + nu * lap_u) */
    #pragma omp parallel for collapse(2) schedule(static)
    for (int j = 1; j < ny - 1; j++) {
        for (int i = 1; i < par->Nx; i++) {  /* u lives on [1, Nx-1] interior */
            double uc = u[IDX(i, j, nx)];

            /* Advection: central differences */
            double dudx = (u[IDX(i+1, j, nx)] - u[IDX(i-1, j, nx)]) * 0.5 * idx;
            double dudy = (u[IDX(i, j+1, nx)] - u[IDX(i, j-1, nx)]) * 0.5 * idy;

            /* v interpolated at u-location */
            double vc = 0.25 * (v[IDX(i, j, nx)]   + v[IDX(i+1, j, nx)] +
                                v[IDX(i, j-1, nx)] + v[IDX(i+1, j-1, nx)]);

            double adv_u = uc * dudx + vc * dudy;

            /* Diffusion: Laplacian */
            double lap_u = (u[IDX(i+1, j, nx)] - 2.0*uc + u[IDX(i-1, j, nx)]) * idx2
                         + (u[IDX(i, j+1, nx)] - 2.0*uc + u[IDX(i, j-1, nx)]) * idy2;

            us[IDX(i, j, nx)] = uc + dt * (-adv_u + nu * lap_u);
        }
    }

    /* v-momentum: v* = v + dt * (-adv_v + nu * lap_v) */
    #pragma omp parallel for collapse(2) schedule(static)
    for (int j = 1; j < ny - 1; j++) {
        for (int i = 1; i < par->Nx + 1; i++) {  /* v interior */
            double vc = v[IDX(i, j, nx)];

            /* Advection */
            double dvdx = (v[IDX(i+1, j, nx)] - v[IDX(i-1, j, nx)]) * 0.5 * idx;
            double dvdy = (v[IDX(i, j+1, nx)] - v[IDX(i, j-1, nx)]) * 0.5 * idy;

            /* u interpolated at v-location */
            double uc = 0.25 * (u[IDX(i, j, nx)]   + u[IDX(i-1, j, nx)] +
                                u[IDX(i, j+1, nx)] + u[IDX(i-1, j+1, nx)]);

            double adv_v = uc * dvdx + vc * dvdy;

            /* Diffusion */
            double lap_v = (v[IDX(i+1, j, nx)] - 2.0*vc + v[IDX(i-1, j, nx)]) * idx2
                         + (v[IDX(i, j+1, nx)] - 2.0*vc + v[IDX(i, j-1, nx)]) * idy2;

            vs[IDX(i, j, nx)] = vc + dt * (-adv_v + nu * lap_v);
        }
    }
}

/* ========================================================================= */
/*  Step 2: Solve Pressure Poisson Equation  ∇²p = (1/dt) ∇·u*              */
/*  Method: Red-Black SOR (parallelizable)                                   */
/* ========================================================================= */

void solve_pressure_poisson(FieldData *fld, const SimParams *par,
                            int *iter_out, double *res_out)
{
    int nx  = fld->nx_loc;
    int ny  = fld->ny_loc;
    double dx  = par->dx;
    double dy  = par->dy;
    double dt  = par->dt;
    double idx = 1.0 / dx;
    double idy = 1.0 / dy;
    double idx2 = 1.0 / (dx * dx);
    double idy2 = 1.0 / (dy * dy);
    double coeff = 1.0 / (2.0 * (idx2 + idy2));
    double omega = par->omega_sor;

    double *p   = fld->p;
    double *us  = fld->u_star;
    double *vs  = fld->v_star;
    double *rhs = fld->rhs;

    /* Compute RHS: (1/dt) * divergence of u* */
    #pragma omp parallel for collapse(2) schedule(static)
    for (int j = 1; j < ny - 1; j++) {
        for (int i = 1; i <= par->Nx; i++) {
            rhs[IDX(i, j, nx)] = (1.0 / dt) * (
                (us[IDX(i, j, nx)] - us[IDX(i-1, j, nx)]) * idx +
                (vs[IDX(i, j, nx)] - vs[IDX(i, j-1, nx)]) * idy
            );
        }
    }

    /* Red-Black SOR iteration */
    int iter;
    double global_res;

    for (iter = 0; iter < par->poisson_maxiter; iter++) {
        
        /* Two sweeps: color=0 (red), color=1 (black) */
        for (int color = 0; color <= 1; color++) {

            /* Exchange pressure ghost rows */
            exchange_ghost_rows(p, par, nx);

            double local_res = 0.0;

            #pragma omp parallel for collapse(2) schedule(static) reduction(+:local_res)
            for (int j = 1; j < ny - 1; j++) {
                for (int i = 1; i <= par->Nx; i++) {
                    /* Red-black coloring based on global indices */
                    int gi = i;
                    int gj = par->j_start + (j - 1);
                    if ((gi + gj) % 2 != color) continue;

                    double p_new = coeff * (
                        idx2 * (p[IDX(i+1, j, nx)] + p[IDX(i-1, j, nx)]) +
                        idy2 * (p[IDX(i, j+1, nx)] + p[IDX(i, j-1, nx)]) -
                        rhs[IDX(i, j, nx)]
                    );

                    /* SOR relaxation */
                    double dp = omega * (p_new - p[IDX(i, j, nx)]);
                    p[IDX(i, j, nx)] += dp;
                    local_res += dp * dp;
                }
            }

            /* Global residual reduction (only on black sweep) */
            if (color == 1) {
                MPI_Allreduce(&local_res, &global_res, 1, MPI_DOUBLE,
                              MPI_SUM, MPI_COMM_WORLD);
                global_res = sqrt(global_res / (double)(par->Nx * par->Ny));
            }
        }

        if (global_res < par->poisson_tol) break;
    }

    /* Apply pressure BC after solve */
    exchange_ghost_rows(p, par, nx);

    /* Neumann BC on physical boundaries */
    #pragma omp parallel for
    for (int j = 0; j < ny; j++) {
        p[IDX(0, j, nx)]            = p[IDX(1, j, nx)];
        p[IDX(par->Nx + 1, j, nx)]  = p[IDX(par->Nx, j, nx)];
    }
    if (par->j_start == 0) {
        #pragma omp parallel for
        for (int i = 0; i < nx; i++)
            p[IDX(i, 0, nx)] = p[IDX(i, 1, nx)];
    }
    if (par->j_end == par->Ny - 1) {
        #pragma omp parallel for
        for (int i = 0; i < nx; i++)
            p[IDX(i, ny - 1, nx)] = p[IDX(i, ny - 2, nx)];
    }

    *iter_out = iter;
    *res_out  = global_res;
}

/* ========================================================================= */
/*  Step 3: Velocity Projection  u^{n+1} = u* - dt * ∇p                     */
/* ========================================================================= */

void project_velocity(FieldData *fld, const SimParams *par)
{
    int nx = fld->nx_loc;
    int ny = fld->ny_loc;
    double dt  = par->dt;
    double idx = 1.0 / par->dx;
    double idy = 1.0 / par->dy;

    double *u  = fld->u;
    double *v  = fld->v;
    double *us = fld->u_star;
    double *vs = fld->v_star;
    double *p  = fld->p;

    /* Update u */
    #pragma omp parallel for collapse(2) schedule(static)
    for (int j = 1; j < ny - 1; j++) {
        for (int i = 1; i < par->Nx; i++) {
            u[IDX(i, j, nx)] = us[IDX(i, j, nx)]
                - dt * (p[IDX(i+1, j, nx)] - p[IDX(i, j, nx)]) * idx;
        }
    }

    /* Update v */
    #pragma omp parallel for collapse(2) schedule(static)
    for (int j = 1; j < ny - 1; j++) {
        for (int i = 1; i <= par->Nx; i++) {
            v[IDX(i, j, nx)] = vs[IDX(i, j, nx)]
                - dt * (p[IDX(i, j+1, nx)] - p[IDX(i, j, nx)]) * idy;
        }
    }
}

/* ========================================================================= */
/*  Diagnostics                                                              */
/* ========================================================================= */

double compute_max_divergence(const FieldData *fld, const SimParams *par)
{
    int nx = fld->nx_loc;
    int ny = fld->ny_loc;
    double idx = 1.0 / par->dx;
    double idy = 1.0 / par->dy;

    double local_max = 0.0;

    #pragma omp parallel for collapse(2) reduction(max:local_max)
    for (int j = 1; j < ny - 1; j++) {
        for (int i = 1; i <= par->Nx; i++) {
            double div = fabs(
                (fld->u[IDX(i, j, nx)] - fld->u[IDX(i-1, j, nx)]) * idx +
                (fld->v[IDX(i, j, nx)] - fld->v[IDX(i, j-1, nx)]) * idy
            );
            if (div > local_max) local_max = div;
        }
    }

    double global_max;
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return global_max;
}

double compute_cfl(const FieldData *fld, const SimParams *par)
{
    int nx = fld->nx_loc;
    int ny = fld->ny_loc;

    double local_max = 0.0;

    #pragma omp parallel for collapse(2) reduction(max:local_max)
    for (int j = 1; j < ny - 1; j++) {
        for (int i = 1; i <= par->Nx; i++) {
            double cfl_val = fabs(fld->u[IDX(i, j, nx)]) / par->dx
                           + fabs(fld->v[IDX(i, j, nx)]) / par->dy;
            cfl_val *= par->dt;
            if (cfl_val > local_max) local_max = cfl_val;
        }
    }

    double global_max;
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return global_max;
}
