/*****************************************************************************
 * main.c - Main driver for 2D Navier-Stokes Lid-Driven Cavity Solver
 *
 * Usage:
 *   mpirun -np 4 ./ns2d_solver -N 256 -Re 1000 -T 20.0 -freq 500
 *
 * Compile:
 *   make
 *****************************************************************************/

#include "ns2d.h"

/* ========================================================================= */
/*  Performance tracking                                                     */
/* ========================================================================= */

void perf_reset(PerfMetrics *pm)
{
    memset(pm, 0, sizeof(PerfMetrics));
}

void perf_report(const PerfMetrics *pm, const SimParams *par)
{
    /* Gather max times across all ranks */
    double times_local[7] = {
        pm->time_advection, pm->time_diffusion, pm->time_pressure,
        pm->time_projection, pm->time_communication, pm->time_io,
        pm->time_total
    };
    double times_max[7];

    MPI_Reduce(times_local, times_max, 7, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (par->rank == 0) {
        printf("\n============================================================\n");
        printf("  Performance Report\n");
        printf("============================================================\n");
        printf("  MPI processes:       %d\n", par->nprocs);
        printf("  OpenMP threads/proc: %d\n", par->nthreads);
        printf("  Total cores:         %d\n", par->nprocs * par->nthreads);
        printf("------------------------------------------------------------\n");
        printf("  Total wall time:     %.3f s\n", times_max[6]);
        printf("  Momentum (adv+diff): %.3f s (%.1f%%)\n",
               times_max[0], 100.0 * times_max[0] / times_max[6]);
        printf("  Pressure Poisson:    %.3f s (%.1f%%)\n",
               times_max[2], 100.0 * times_max[2] / times_max[6]);
        printf("  Projection:          %.3f s (%.1f%%)\n",
               times_max[3], 100.0 * times_max[3] / times_max[6]);
        printf("  I/O:                 %.3f s (%.1f%%)\n",
               times_max[5], 100.0 * times_max[5] / times_max[6]);
        printf("------------------------------------------------------------\n");
        double cells_per_sec = (double)par->Nx * par->Ny / times_max[6];
        printf("  Throughput:          %.2e cell-updates/s\n", cells_per_sec);
        printf("============================================================\n");
    }
}

/* ========================================================================= */
/*  Main                                                                     */
/* ========================================================================= */

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    SimParams par;
    FieldData fld;
    PerfMetrics perf;

    memset(&par, 0, sizeof(SimParams));
    memset(&fld, 0, sizeof(FieldData));

    /* Initialize parameters and domain decomposition */
    params_init(&par, argc, argv);

    /* Allocate and initialize fields */
    field_alloc(&fld, &par);
    field_init(&fld, &par);

    /* Create output directory */
    if (par.rank == 0) {
        char cmd[512];
        snprintf(cmd, sizeof(cmd), "mkdir -p %s", par.output_dir);
        system(cmd);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    /* Performance tracking */
    perf_reset(&perf);
    double t0_total = MPI_Wtime();

    /* ================================================================= */
    /*  Time-stepping loop                                               */
    /* ================================================================= */

    double time = 0.0;
    int step = 0;
    int converged = 0;

    /* Write initial condition */
    double t0_io = MPI_Wtime();
    write_vtk(&fld, &par, 0);
    write_centerline(&fld, &par, 0);
    perf.time_io += MPI_Wtime() - t0_io;

    while (time < par.T_final && step < par.max_steps) {
        step++;
        time += par.dt;

        /* ---- Step 1: Compute intermediate velocity ---- */
        double t0 = MPI_Wtime();
        compute_rhs_momentum(&fld, &par);
        perf.time_advection += MPI_Wtime() - t0;

        /* ---- Step 2: Solve pressure Poisson equation ---- */
        int p_iter;
        double p_res;
        t0 = MPI_Wtime();
        solve_pressure_poisson(&fld, &par, &p_iter, &p_res);
        perf.time_pressure += MPI_Wtime() - t0;

        /* ---- Step 3: Project velocity to be divergence-free ---- */
        t0 = MPI_Wtime();
        project_velocity(&fld, &par);
        perf.time_projection += MPI_Wtime() - t0;

        /* ---- Apply boundary conditions ---- */
        apply_boundary_conditions(&fld, &par);

        /* ---- Diagnostics ---- */
        if (step % 100 == 0 || step == 1) {
            double max_div = compute_max_divergence(&fld, &par);
            double cfl     = compute_cfl(&fld, &par);

            if (par.rank == 0) {
                printf("  Step %6d | t=%.4f | CFL=%.4f | div=%.2e | "
                       "P_iter=%4d P_res=%.2e\n",
                       step, time, cfl, max_div, p_iter, p_res);
            }

            /* Check for steady state */
            if (max_div < 1.0e-10 && step > 1000) {
                converged = 1;
            }
        }

        /* ---- Output ---- */
        if (step % par.output_freq == 0) {
            t0 = MPI_Wtime();
            write_vtk(&fld, &par, step);
            write_centerline(&fld, &par, step);
            perf.time_io += MPI_Wtime() - t0;
        }
    }

    /* Write final state */
    double t0_io2 = MPI_Wtime();
    write_vtk(&fld, &par, step);
    write_centerline(&fld, &par, step);
    perf.time_io += MPI_Wtime() - t0_io2;

    perf.time_total = MPI_Wtime() - t0_total;

    /* Performance report */
    perf_report(&perf, &par);

    if (par.rank == 0) {
        printf("\nSimulation %s after %d steps (t=%.4f)\n",
               converged ? "CONVERGED" : "completed", step, time);
    }

    /* Cleanup */
    field_free(&fld);
    MPI_Finalize();

    return 0;
}
