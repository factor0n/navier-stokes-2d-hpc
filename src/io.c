/*****************************************************************************
 * io.c - Output routines: VTK files and centerline velocity profiles
 *
 * VTK output gathers data on rank 0 and writes structured grid files
 * compatible with ParaView.
 *****************************************************************************/

#include "ns2d.h"

/* ========================================================================= */
/*  Write VTK Structured Grid (gathered on rank 0)                           */
/* ========================================================================= */

void write_vtk(const FieldData *fld, const SimParams *par, int step)
{
    int Nx = par->Nx;
    int Ny = par->Ny;
    int nx = fld->nx_loc;
    int local_ny = par->local_ny;

    /* Interpolate u, v to cell centers for output */
    int n_local = Nx * local_ny;
    double *uc_local = (double *)malloc(n_local * sizeof(double));
    double *vc_local = (double *)malloc(n_local * sizeof(double));
    double *pc_local = (double *)malloc(n_local * sizeof(double));

    #pragma omp parallel for collapse(2)
    for (int jj = 0; jj < local_ny; jj++) {
        for (int ii = 0; ii < Nx; ii++) {
            int j = jj + 1;  /* Skip ghost */
            int i = ii + 1;
            int idx_loc = jj * Nx + ii;

            /* Interpolate u to cell center */
            uc_local[idx_loc] = 0.5 * (fld->u[IDX(i, j, nx)] + fld->u[IDX(i-1, j, nx)]);
            /* Interpolate v to cell center */
            vc_local[idx_loc] = 0.5 * (fld->v[IDX(i, j, nx)] + fld->v[IDX(i, j-1, nx)]);
            /* Pressure already cell-centered */
            pc_local[idx_loc] = fld->p[IDX(i, j, nx)];
        }
    }

    /* Gather on rank 0 */
    double *uc_global = NULL;
    double *vc_global = NULL;
    double *pc_global = NULL;

    int *recvcounts = NULL;
    int *displs = NULL;

    if (par->rank == 0) {
        int n_global = Nx * Ny;
        uc_global = (double *)malloc(n_global * sizeof(double));
        vc_global = (double *)malloc(n_global * sizeof(double));
        pc_global = (double *)malloc(n_global * sizeof(double));

        recvcounts = (int *)malloc(par->nprocs * sizeof(int));
        displs     = (int *)malloc(par->nprocs * sizeof(int));
    }

    /* Gather recv counts */
    int sendcount = n_local;
    MPI_Gather(&sendcount, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (par->rank == 0) {
        displs[0] = 0;
        for (int r = 1; r < par->nprocs; r++) {
            displs[r] = displs[r-1] + recvcounts[r-1];
        }
    }

    MPI_Gatherv(uc_local, sendcount, MPI_DOUBLE,
                uc_global, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(vc_local, sendcount, MPI_DOUBLE,
                vc_global, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(pc_local, sendcount, MPI_DOUBLE,
                pc_global, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Write VTK file on rank 0 */
    if (par->rank == 0) {
        char filename[512];
        snprintf(filename, sizeof(filename), "%s/cavity_%06d.vtk",
                 par->output_dir, step);

        FILE *fp = fopen(filename, "w");
        if (!fp) {
            fprintf(stderr, "Error: Cannot open %s for writing\n", filename);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        fprintf(fp, "# vtk DataFile Version 3.0\n");
        fprintf(fp, "Lid-Driven Cavity Re=%.0f Step=%d\n", par->Re, step);
        fprintf(fp, "ASCII\n");
        fprintf(fp, "DATASET STRUCTURED_POINTS\n");
        fprintf(fp, "DIMENSIONS %d %d 1\n", Nx, Ny);
        fprintf(fp, "ORIGIN %.6f %.6f 0.0\n", 0.5 * par->dx, 0.5 * par->dy);
        fprintf(fp, "SPACING %.6f %.6f 1.0\n", par->dx, par->dy);

        fprintf(fp, "POINT_DATA %d\n", Nx * Ny);

        /* Velocity vector */
        fprintf(fp, "VECTORS velocity double\n");
        for (int idx = 0; idx < Nx * Ny; idx++) {
            fprintf(fp, "%.8e %.8e 0.0\n", uc_global[idx], vc_global[idx]);
        }

        /* Velocity magnitude */
        fprintf(fp, "SCALARS velocity_magnitude double 1\n");
        fprintf(fp, "LOOKUP_TABLE default\n");
        for (int idx = 0; idx < Nx * Ny; idx++) {
            double mag = sqrt(uc_global[idx]*uc_global[idx] + 
                              vc_global[idx]*vc_global[idx]);
            fprintf(fp, "%.8e\n", mag);
        }

        /* Pressure */
        fprintf(fp, "SCALARS pressure double 1\n");
        fprintf(fp, "LOOKUP_TABLE default\n");
        for (int idx = 0; idx < Nx * Ny; idx++) {
            fprintf(fp, "%.8e\n", pc_global[idx]);
        }

        /* Vorticity (computed from global fields) */
        fprintf(fp, "SCALARS vorticity double 1\n");
        fprintf(fp, "LOOKUP_TABLE default\n");
        for (int j = 0; j < Ny; j++) {
            for (int i = 0; i < Nx; i++) {
                double dvdx, dudy;
                int im = (i > 0)      ? i - 1 : i;
                int ip = (i < Nx - 1) ? i + 1 : i;
                int jm = (j > 0)      ? j - 1 : j;
                int jp = (j < Ny - 1) ? j + 1 : j;

                dvdx = (vc_global[j * Nx + ip] - vc_global[j * Nx + im])
                       / ((ip - im) * par->dx);
                dudy = (uc_global[jp * Nx + i] - uc_global[jm * Nx + i])
                       / ((jp - jm) * par->dy);

                fprintf(fp, "%.8e\n", dvdx - dudy);
            }
        }

        fclose(fp);
        printf("  [I/O] Wrote %s\n", filename);

        free(uc_global);
        free(vc_global);
        free(pc_global);
        free(recvcounts);
        free(displs);
    }

    free(uc_local);
    free(vc_local);
    free(pc_local);
}

/* ========================================================================= */
/*  Write centerline velocity profiles (for validation against Ghia et al.)  */
/* ========================================================================= */

void write_centerline(const FieldData *fld, const SimParams *par, int step)
{
    int Nx = par->Nx;
    int Ny = par->Ny;
    int nx = fld->nx_loc;
    int local_ny = par->local_ny;

    /* Extract u along vertical centerline (x = 0.5) */
    int i_center = Nx / 2 + 1;  /* Including ghost offset */

    double *u_center_local = (double *)calloc(local_ny, sizeof(double));
    double *v_center_local = (double *)calloc(local_ny, sizeof(double));

    for (int jj = 0; jj < local_ny; jj++) {
        int j = jj + 1;
        /* Interpolate to cell center */
        u_center_local[jj] = 0.5 * (fld->u[IDX(i_center, j, nx)] +
                                     fld->u[IDX(i_center - 1, j, nx)]);
        v_center_local[jj] = 0.5 * (fld->v[IDX(i_center, j, nx)] +
                                     fld->v[IDX(i_center, j - 1, nx)]);
    }

    /* Gather on rank 0 */
    double *u_center = NULL;
    double *v_center = NULL;
    int *recvcounts = NULL;
    int *displs = NULL;

    if (par->rank == 0) {
        u_center   = (double *)malloc(Ny * sizeof(double));
        v_center   = (double *)malloc(Ny * sizeof(double));
        recvcounts = (int *)malloc(par->nprocs * sizeof(int));
        displs     = (int *)malloc(par->nprocs * sizeof(int));
    }

    int sendcount = local_ny;
    MPI_Gather(&sendcount, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (par->rank == 0) {
        displs[0] = 0;
        for (int r = 1; r < par->nprocs; r++)
            displs[r] = displs[r-1] + recvcounts[r-1];
    }

    MPI_Gatherv(u_center_local, sendcount, MPI_DOUBLE,
                u_center, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(v_center_local, sendcount, MPI_DOUBLE,
                v_center, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (par->rank == 0) {
        char filename[512];
        snprintf(filename, sizeof(filename), "%s/centerline_%06d.csv",
                 par->output_dir, step);

        FILE *fp = fopen(filename, "w");
        fprintf(fp, "y,u_centerline,v_centerline\n");
        for (int j = 0; j < Ny; j++) {
            double y = (j + 0.5) * par->dy;
            fprintf(fp, "%.8e,%.8e,%.8e\n", y, u_center[j], v_center[j]);
        }
        fclose(fp);

        free(u_center);
        free(v_center);
        free(recvcounts);
        free(displs);
    }

    free(u_center_local);
    free(v_center_local);
}
