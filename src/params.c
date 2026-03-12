/*****************************************************************************
 * params.c - Parameter initialization and MPI domain decomposition
 *****************************************************************************/

#include "ns2d.h"

void params_init(SimParams *par, int argc, char **argv)
{
    MPI_Comm_rank(MPI_COMM_WORLD, &par->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &par->nprocs);

    /* Default parameters */
    par->Nx    = 256;
    par->Ny    = 256;
    par->Lx    = 1.0;
    par->Ly    = 1.0;
    par->Re    = 1000.0;
    par->U_lid = 1.0;
    par->T_final   = 20.0;
    par->max_steps = 100000;

    par->poisson_maxiter = 5000;
    par->poisson_tol     = 1.0e-6;
    par->omega_sor       = 1.7;

    par->output_freq = 500;
    strcpy(par->output_dir, "results");

    /* Parse command-line arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-N") == 0 && i + 1 < argc) {
            par->Nx = par->Ny = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-Nx") == 0 && i + 1 < argc) {
            par->Nx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-Ny") == 0 && i + 1 < argc) {
            par->Ny = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-Re") == 0 && i + 1 < argc) {
            par->Re = atof(argv[++i]);
        } else if (strcmp(argv[i], "-T") == 0 && i + 1 < argc) {
            par->T_final = atof(argv[++i]);
        } else if (strcmp(argv[i], "-dt") == 0 && i + 1 < argc) {
            par->dt = atof(argv[++i]);
        } else if (strcmp(argv[i], "-freq") == 0 && i + 1 < argc) {
            par->output_freq = atoi(argv[++i]);
        }
    }

    /* Derived quantities */
    par->dx = par->Lx / (double)par->Nx;
    par->dy = par->Ly / (double)par->Ny;
    par->nu = par->U_lid * par->Lx / par->Re;

    /* Stability: CFL and diffusive constraints */
    double dt_cfl  = 0.25 * fmin(par->dx, par->dy) / par->U_lid;
    double dt_diff = 0.25 * par->dx * par->dy / par->nu;
    double dt_auto = 0.5 * fmin(dt_cfl, dt_diff);

    /* Use user-specified dt if given, otherwise auto */
    if (par->dt <= 0.0 || par->dt > dt_auto) {
        par->dt = dt_auto;
    }

    /* OpenMP threads */
    #pragma omp parallel
    {
        #pragma omp single
        par->nthreads = omp_get_num_threads();
    }

    /* ===================================================================== */
    /*  1D Domain Decomposition in y-direction                               */
    /* ===================================================================== */
    int base_rows = par->Ny / par->nprocs;
    int remainder = par->Ny % par->nprocs;

    if (par->rank < remainder) {
        par->local_ny = base_rows + 1;
        par->j_start  = par->rank * (base_rows + 1);
    } else {
        par->local_ny = base_rows;
        par->j_start  = remainder * (base_rows + 1) 
                       + (par->rank - remainder) * base_rows;
    }
    par->j_end = par->j_start + par->local_ny - 1;

    /* Print configuration */
    if (par->rank == 0) {
        printf("============================================================\n");
        printf("  2D Navier-Stokes Solver - Lid-Driven Cavity\n");
        printf("  Hybrid MPI (%d procs) + OpenMP (%d threads/proc)\n",
               par->nprocs, par->nthreads);
        printf("============================================================\n");
        printf("  Grid:       %d x %d\n", par->Nx, par->Ny);
        printf("  Reynolds:   %.1f\n", par->Re);
        printf("  dt:         %.6e  (CFL=%.3e, Diff=%.3e)\n",
               par->dt, dt_cfl, dt_diff);
        printf("  T_final:    %.2f\n", par->T_final);
        printf("  Poisson:    maxiter=%d, tol=%.1e, omega=%.2f\n",
               par->poisson_maxiter, par->poisson_tol, par->omega_sor);
        printf("============================================================\n");
    }

    for (int r = 0; r < par->nprocs; r++) {
        if (par->rank == r) {
            printf("  Rank %3d: rows [%4d, %4d] (%d rows)\n",
                   par->rank, par->j_start, par->j_end, par->local_ny);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    if (par->rank == 0) {
        printf("============================================================\n\n");
    }
}
