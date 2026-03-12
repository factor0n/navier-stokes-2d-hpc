###############################################################################
# Makefile - 2D Navier-Stokes Solver (MPI + OpenMP)
###############################################################################

CC       = mpicc
CFLAGS   = -O3 -march=native -fopenmp -Wall -Wextra -std=c99
LDFLAGS  = -fopenmp -lm

# Directories
SRCDIR   = src
INCDIR   = include
BUILDDIR = build
TARGET   = ns2d_solver

# Sources and objects
SRCS     = $(wildcard $(SRCDIR)/*.c)
OBJS     = $(patsubst $(SRCDIR)/%.c, $(BUILDDIR)/%.o, $(SRCS))

# Include path
CFLAGS  += -I$(INCDIR)

###############################################################################
# Targets
###############################################################################

.PHONY: all clean run run-small benchmark help

all: $(BUILDDIR) $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $@ $(LDFLAGS)
	@echo "============================================"
	@echo "  Build successful: $(TARGET)"
	@echo "============================================"

$(BUILDDIR)/%.o: $(SRCDIR)/%.c $(INCDIR)/ns2d.h
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

clean:
	rm -rf $(BUILDDIR) $(TARGET) results/*.vtk results/*.csv

# ---- Quick test runs ----

run-small:
	@echo "Running small test (64x64, Re=100, T=2.0)..."
	OMP_NUM_THREADS=2 mpirun --oversubscribe -np 2 ./$(TARGET) \
		-N 64 -Re 100 -T 2.0 -freq 100

run:
	@echo "Running standard case (256x256, Re=1000, T=20.0)..."
	OMP_NUM_THREADS=2 mpirun --oversubscribe -np 4 ./$(TARGET) \
		-N 256 -Re 1000 -T 20.0 -freq 500

run-high-re:
	@echo "Running high Reynolds (256x256, Re=5000, T=50.0)..."
	OMP_NUM_THREADS=2 mpirun --oversubscribe -np 4 ./$(TARGET) \
		-N 256 -Re 5000 -T 50.0 -freq 1000

# ---- Scaling benchmark ----

benchmark:
	@echo "===== Scaling Benchmark (128x128, Re=400, T=1.0) ====="
	@echo ""
	@for np in 1 2 4; do \
		echo "--- MPI procs = $$np ---"; \
		OMP_NUM_THREADS=1 mpirun --oversubscribe -np $$np ./$(TARGET) \
			-N 128 -Re 400 -T 1.0 -freq 999999 2>&1 | \
			grep -E "(Rank|Total wall|Throughput|Performance)"; \
		echo ""; \
	done
	@echo "===== OpenMP Scaling (1 MPI proc, N=128) ====="
	@for nt in 1 2 4; do \
		echo "--- OpenMP threads = $$nt ---"; \
		OMP_NUM_THREADS=$$nt mpirun --oversubscribe -np 1 ./$(TARGET) \
			-N 128 -Re 400 -T 1.0 -freq 999999 2>&1 | \
			grep -E "(threads|Total wall|Throughput)"; \
		echo ""; \
	done

# ---- Help ----

help:
	@echo "Targets:"
	@echo "  all        - Build the solver"
	@echo "  clean      - Remove build files"
	@echo "  run-small  - Quick test (64x64, Re=100)"
	@echo "  run        - Standard run (256x256, Re=1000)"
	@echo "  run-high-re - High Reynolds (256x256, Re=5000)"
	@echo "  benchmark  - Scaling benchmark"
	@echo ""
	@echo "Parameters:"
	@echo "  -N <int>    Grid size (NxN)"
	@echo "  -Nx <int>   Grid points in x"
	@echo "  -Ny <int>   Grid points in y"
	@echo "  -Re <float> Reynolds number"
	@echo "  -T <float>  Final time"
	@echo "  -dt <float> Time step (auto if not set)"
	@echo "  -freq <int> Output frequency"
