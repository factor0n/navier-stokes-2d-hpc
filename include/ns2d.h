/*****************************************************************************
 * ns2d.h - 2D Navier-Stokes Solver (Lid-Driven Cavity)
 * 
 * Hybrid MPI + OpenMP parallelization
 * Fractional Step Method (Chorin's Projection)
 * Finite Differences on a Staggered Grid (MAC)
 *
 * Author: [ayoub gounnou]
 * Date:   2026
 * License: MIT
 *****************************************************************************/

#ifndef NS2D_H
#define NS2D_H

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>

/* ========================================================================= */
/*  Simulation Parameters                                                    */
/* ========================================================================= */

typedef struct {
    /* Grid */
    int    Nx;          /* Global grid points in x                           */
    int    Ny;          /* Global grid points in y                           */
    double Lx;          /* Domain length in x                                */
    double Ly;          /* Domain length in y                                */
    double dx;          /* Grid spacing in x                                 */
    double dy;          /* Grid spacing in y                                 */

    /* Physics */
    double Re;          /* Reynolds number                                   */
    double nu;          /* Kinematic viscosity (= 1/Re for unit cavity)      */
    double U_lid;       /* Lid velocity                                      */

    /* Time */
    double dt;          /* Time step                                         */
    double T_final;     /* Final time                                        */
    int    max_steps;   /* Maximum number of time steps                      */

    /* Pressure solver */
    int    poisson_maxiter;  /* Max Poisson iterations                       */
    double poisson_tol;      /* Poisson convergence tolerance                */
    double omega_sor;        /* SOR relaxation parameter                     */

    /* Output */
    int    output_freq;      /* Output every N steps                         */
    char   output_dir[256];  /* Output directory                             */

    /* MPI */
    int    rank;        /* MPI rank                                          */
    int    nprocs;      /* Number of MPI processes                           */
    int    local_ny;    /* Local number of rows (y-direction decomposition)  */
    int    j_start;     /* Global starting j-index for this rank             */
    int    j_end;       /* Global ending j-index for this rank               */

    /* OpenMP */
    int    nthreads;    /* Number of OpenMP threads per process              */

} SimParams;

/* ========================================================================= */
/*  Field Data (local to each MPI rank)                                      */
/* ========================================================================= */

typedef struct {
    /* Velocity components (staggered: u at (i+1/2,j), v at (i,j+1/2)) */
    double *u;          /* x-velocity  [Nx+2] x [local_ny+2] with ghosts    */
    double *v;          /* y-velocity  [Nx+2] x [local_ny+2] with ghosts    */

    /* Pressure (cell-centered) */
    double *p;          /* Pressure    [Nx+2] x [local_ny+2] with ghosts    */

    /* Intermediate velocity (predictor step) */
    double *u_star;
    double *v_star;

    /* Right-hand side of pressure Poisson equation */
    double *rhs;

    /* Dimensions including ghost cells */
    int nx_loc;         /* = Nx + 2 (includes ghost layers in x)            */
    int ny_loc;         /* = local_ny + 2 (includes ghost layers in y)      */

} FieldData;

/* ========================================================================= */
/*  Performance Metrics                                                      */
/* ========================================================================= */

typedef struct {
    double time_advection;
    double time_diffusion;
    double time_pressure;
    double time_projection;
    double time_communication;
    double time_io;
    double time_total;
} PerfMetrics;

/* ========================================================================= */
/*  Macros for 2D array indexing (row-major)                                 */
/* ========================================================================= */

#define IDX(i, j, nx) ((j) * (nx) + (i))

/* ========================================================================= */
/*  Function Prototypes                                                      */
/* ========================================================================= */

/* Initialization */
void params_init(SimParams *par, int argc, char **argv);
void field_alloc(FieldData *fld, const SimParams *par);
void field_free(FieldData *fld);
void field_init(FieldData *fld, const SimParams *par);

/* Boundary conditions */
void apply_boundary_conditions(FieldData *fld, const SimParams *par);

/* MPI communication */
void exchange_ghost_rows(double *field, const SimParams *par, int nx_loc);

/* Time stepping (Chorin's projection) */
void compute_rhs_momentum(FieldData *fld, const SimParams *par);
void solve_pressure_poisson(FieldData *fld, const SimParams *par,
                            int *iter_out, double *res_out);
void project_velocity(FieldData *fld, const SimParams *par);

/* I/O */
void write_vtk(const FieldData *fld, const SimParams *par, int step);
void write_centerline(const FieldData *fld, const SimParams *par, int step);

/* Performance */
void perf_reset(PerfMetrics *pm);
void perf_report(const PerfMetrics *pm, const SimParams *par);

/* Utilities */
double compute_max_divergence(const FieldData *fld, const SimParams *par);
double compute_cfl(const FieldData *fld, const SimParams *par);

#endif /* NS2D_H */
