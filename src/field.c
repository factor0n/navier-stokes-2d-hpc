/*****************************************************************************
 * field.c - Field allocation, initialization, and boundary conditions
 *****************************************************************************/

#include "ns2d.h"

void field_alloc(FieldData *fld, const SimParams *par)
{
    fld->nx_loc = par->Nx + 2;       /* +2 for ghost cells in x */
    fld->ny_loc = par->local_ny + 2; /* +2 for ghost cells in y */

    size_t n = (size_t)fld->nx_loc * (size_t)fld->ny_loc;

    fld->u      = (double *)calloc(n, sizeof(double));
    fld->v      = (double *)calloc(n, sizeof(double));
    fld->p      = (double *)calloc(n, sizeof(double));
    fld->u_star = (double *)calloc(n, sizeof(double));
    fld->v_star = (double *)calloc(n, sizeof(double));
    fld->rhs    = (double *)calloc(n, sizeof(double));

    if (!fld->u || !fld->v || !fld->p || 
        !fld->u_star || !fld->v_star || !fld->rhs) {
        fprintf(stderr, "Rank %d: Memory allocation failed!\n", par->rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

void field_free(FieldData *fld)
{
    free(fld->u);
    free(fld->v);
    free(fld->p);
    free(fld->u_star);
    free(fld->v_star);
    free(fld->rhs);
}

void field_init(FieldData *fld, const SimParams *par)
{
    int nx = fld->nx_loc;
    int ny = fld->ny_loc;

    /* Zero everywhere (already done by calloc, but explicit) */
    #pragma omp parallel for collapse(2)
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            fld->u[IDX(i, j, nx)] = 0.0;
            fld->v[IDX(i, j, nx)] = 0.0;
            fld->p[IDX(i, j, nx)] = 0.0;
        }
    }

    /* Apply initial boundary conditions */
    apply_boundary_conditions(fld, par);
}

/* ========================================================================= */
/*  Boundary Conditions for Lid-Driven Cavity                                */
/*  - Top wall (y=Ly):   u = U_lid, v = 0                                   */
/*  - Other walls:        u = 0,     v = 0 (no-slip)                         */
/* ========================================================================= */

void apply_boundary_conditions(FieldData *fld, const SimParams *par)
{
    int nx = fld->nx_loc;
    int ny = fld->ny_loc;
    int Nx = par->Nx;
    double U_lid = par->U_lid;

    /* ---- Left wall (i=0) and Right wall (i=Nx+1) ---- */
    #pragma omp parallel for
    for (int j = 0; j < ny; j++) {
        /* Left: no-slip */
        fld->u[IDX(0, j, nx)]      = 0.0;
        fld->v[IDX(0, j, nx)]      = -fld->v[IDX(1, j, nx)];

        /* Right: no-slip */
        fld->u[IDX(Nx, j, nx)]     = 0.0;
        fld->v[IDX(Nx + 1, j, nx)] = -fld->v[IDX(Nx, j, nx)];
    }

    /* ---- Bottom wall: only if this rank owns j_start == 0 ---- */
    if (par->j_start == 0) {
        #pragma omp parallel for
        for (int i = 0; i < nx; i++) {
            fld->u[IDX(i, 0, nx)] = -fld->u[IDX(i, 1, nx)];
            fld->v[IDX(i, 0, nx)] = 0.0;
        }
    }

    /* ---- Top wall (lid): only if this rank owns j_end == Ny-1 ---- */
    if (par->j_end == par->Ny - 1) {
        int j_top = ny - 1;  /* Ghost row at top */
        #pragma omp parallel for
        for (int i = 0; i < nx; i++) {
            fld->u[IDX(i, j_top, nx)] = 2.0 * U_lid - fld->u[IDX(i, j_top - 1, nx)];
            fld->v[IDX(i, j_top - 1, nx)] = 0.0;
        }
    }

    /* ---- Pressure: Neumann BC (zero normal gradient) ---- */
    /* Left & Right */
    #pragma omp parallel for
    for (int j = 0; j < ny; j++) {
        fld->p[IDX(0, j, nx)]      = fld->p[IDX(1, j, nx)];
        fld->p[IDX(Nx + 1, j, nx)] = fld->p[IDX(Nx, j, nx)];
    }

    /* Bottom */
    if (par->j_start == 0) {
        #pragma omp parallel for
        for (int i = 0; i < nx; i++) {
            fld->p[IDX(i, 0, nx)] = fld->p[IDX(i, 1, nx)];
        }
    }

    /* Top */
    if (par->j_end == par->Ny - 1) {
        int j_top = ny - 1;
        #pragma omp parallel for
        for (int i = 0; i < nx; i++) {
            fld->p[IDX(i, j_top, nx)] = fld->p[IDX(i, j_top - 1, nx)];
        }
    }
}
