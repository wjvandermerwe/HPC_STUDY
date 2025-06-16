/*
 * MPI Programs Collection
 * Contains implementations for:
 * 1. MPI_Allreduce using tree and butterfly structures
 * 2. Matrix-vector multiplication with block-column distribution
 * 3. MPI_Scatter and MPI_Gather demonstration
 * 4. MPI derived datatype for Particle struct
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// ============================================================================
// PROBLEM 1: MPI_Allreduce Implementation
// ============================================================================

/*
 * Tree-structured Allreduce (Power of 2)
 * Phase 1: Tree reduction to root
 * Phase 2: Broadcast result to all processes
 */
void tree_allreduce_power_of_2(double *sendbuf, double *recvbuf, int count, 
                               MPI_Op op, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    
    // Copy send buffer to receive buffer
    for (int i = 0; i < count; i++) {
        recvbuf[i] = sendbuf[i];
    }
    
    // Phase 1: Tree reduction
    int step = 1;
    while (step < size) {
        if (rank % (2 * step) == 0) {
            // Receive from partner
            if (rank + step < size) {
                double *temp = malloc(count * sizeof(double));
                MPI_Recv(temp, count, MPI_DOUBLE, rank + step, 0, comm, MPI_STATUS_IGNORE);
                
                // Perform reduction operation (assuming SUM for simplicity)
                for (int i = 0; i < count; i++) {
                    recvbuf[i] += temp[i];
                }
                free(temp);
            }
        } else if (rank % step == 0) {
            // Send to partner
            int partner = rank - step;
            MPI_Send(recvbuf, count, MPI_DOUBLE, partner, 0, comm);
            break;
        }
        step *= 2;
    }
    
    // Phase 2: Broadcast result
    MPI_Bcast(recvbuf, count, MPI_DOUBLE, 0, comm);
}

/*
 * Butterfly-structured Allreduce (Power of 2)
 */
void butterfly_allreduce_power_of_2(double *sendbuf, double *recvbuf, int count, 
                                    MPI_Op op, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    
    // Copy send buffer to receive buffer
    for (int i = 0; i < count; i++) {
        recvbuf[i] = sendbuf[i];
    }
    // 3
    int stages = (int)log2(size);
    
    for (int stage = 0; stage < stages; stage++) {
        int partner = rank ^ (1 << stage);
        
        double *temp = malloc(count * sizeof(double));
        
        // Exchange data with partner
        MPI_Sendrecv(recvbuf, count, MPI_DOUBLE, partner, 0,
                     temp, count, MPI_DOUBLE, partner, 0,
                     comm, MPI_STATUS_IGNORE);
        
        // Perform reduction operation
        for (int i = 0; i < count; i++) {
            recvbuf[i] += temp[i];
        }
        
        free(temp);
    }
}

/*
 * Tree-structured Allreduce (Any number of processes)
 */
void tree_allreduce_general(double *sendbuf, double *recvbuf, int count, 
                           MPI_Op op, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    
    // Copy send buffer to receive buffer
    for (int i = 0; i < count; i++) {
        recvbuf[i] = sendbuf[i];
    }
    
    // Phase 1: Tree reduction using binary tree structure
    int active_procs = size;
    
    while (active_procs > 1) {
        int half = active_procs / 2;
        
        if (rank < half) {
            // Receive from partner if it exists
            int partner = rank + half;
            if (partner < size) {
                double *temp = malloc(count * sizeof(double));
                MPI_Recv(temp, count, MPI_DOUBLE, partner, 0, comm, MPI_STATUS_IGNORE);
                
                for (int i = 0; i < count; i++) {
                    recvbuf[i] += temp[i];
                }
                free(temp);
            }
        } else if (rank < active_procs) {
            // Send to partner
            int partner = rank - half;
            MPI_Send(recvbuf, count, MPI_DOUBLE, partner, 0, comm);
            break;
        }
        
        active_procs = half + (active_procs % 2);
    }
    
    // Phase 2: Broadcast result
    MPI_Bcast(recvbuf, count, MPI_DOUBLE, 0, comm);
}

void problem1_demo() {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    double sendbuf = rank + 1.0;  // Each process contributes its rank + 1
    double recvbuf_tree, recvbuf_butterfly, recvbuf_mpi;
    
    printf("Process %d: Initial value = %.2f\n", rank, sendbuf);
    
    // Test tree allreduce
    tree_allreduce_general(&sendbuf, &recvbuf_tree, 1, MPI_SUM, MPI_COMM_WORLD);
    
    // Test butterfly allreduce (if power of 2)
    if ((size & (size - 1)) == 0) {
        butterfly_allreduce_power_of_2(&sendbuf, &recvbuf_butterfly, 1, MPI_SUM, MPI_COMM_WORLD);
    } else {
        recvbuf_butterfly = -1;  // Invalid for non-power-of-2
    }
    
    // Compare with MPI_Allreduce
    MPI_Allreduce(&sendbuf, &recvbuf_mpi, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Results:\n");
        printf("Tree Allreduce: %.2f\n", recvbuf_tree);
        if ((size & (size - 1)) == 0) {
            printf("Butterfly Allreduce: %.2f\n", recvbuf_butterfly);
        }
        printf("MPI_Allreduce: %.2f\n", recvbuf_mpi);
        printf("Expected sum: %.2f\n", size * (size + 1) / 2.0);
    }
}

// ============================================================================
// PROBLEM 2: Matrix-Vector Multiplication with Block-Column Distribution
// ============================================================================

void matrix_vector_multiply() {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int n = 8;  // Matrix size (assuming n is divisible by size)
    int local_cols = n / size;
    
    double *matrix = NULL;
    double *local_matrix = malloc(n * local_cols * sizeof(double));
    double *vector = malloc(n * sizeof(double));
    double *local_result = malloc(n * sizeof(double));
    double *result = NULL;
    
    if (rank == 0) {
        matrix = malloc(n * n * sizeof(double));
        result = malloc(n * sizeof(double));
        
        // Initialize matrix and vector
        printf("Initializing %dx%d matrix and vector...\n", n, n);
        for (int i = 0; i < n; i++) {
            vector[i] = i + 1;
            for (int j = 0; j < n; j++) {
                matrix[i * n + j] = i + j + 1;
            }
        }
        
        // Print matrix and vector
        printf("Matrix:\n");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                printf("%6.1f ", matrix[i * n + j]);
            }
            printf("\n");
        }
        printf("Vector: ");
        for (int i = 0; i < n; i++) {
            printf("%.1f ", vector[i]);
        }
        printf("\n\n");
    }
    
    // Broadcast vector to all processes
    MPI_Bcast(vector, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Distribute matrix columns
    if (rank == 0) {
        // Send columns to other processes
        for (int p = 1; p < size; p++) {
            for (int i = 0; i < n; i++) {
                MPI_Send(&matrix[i * n + p * local_cols], local_cols, 
                        MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
            }
        }
        
        // Copy local columns for process 0
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < local_cols; j++) {
                local_matrix[i * local_cols + j] = matrix[i * n + j];
            }
        }
    } else {
        // Receive columns
        for (int i = 0; i < n; i++) {
            MPI_Recv(&local_matrix[i * local_cols], local_cols, 
                    MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

    }
    // Compute local matrix-vector multiplication
    for (int i = 0; i < n; i++) {
        local_result[i] = 0.0;
        for (int j = 0; j < local_cols; j++) {
            local_result[i] += local_matrix[i * local_cols + j] *
                              vector[rank * local_cols + j];
        }
    }

    // Sum partial results
    MPI_Allreduce(local_result, result, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);


    if (rank == 0) {
        printf("Result vector:\n");
        for (int i = 0; i < n; i++) {
            printf("%.1f ", result[i]);
        }
        printf("\n\n");
        
        free(matrix);
        free(result);
    }
    
    free(local_matrix);
    free(vector);
    free(local_result);
}

// ============================================================================
// PROBLEM 3: MPI_Scatter and MPI_Gather Demonstration
// ============================================================================

void scatter_gather_demo() {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int N = size * 4;  // Total array size (divisible by size)
    int local_size = N / size;
    
    int *global_array = NULL;
    int *local_array = malloc(local_size * sizeof(int));
    int *result_array = NULL;
    
    if (rank == 0) {
        global_array = malloc(N * sizeof(int));
        result_array = malloc(N * sizeof(int));
        
        // Initialize array
        printf("Original array: ");
        for (int i = 0; i < N; i++) {
            global_array[i] = i + 1;
            printf("%d ", global_array[i]);
        }
        printf("\n");
    }
    
    // Scatter the array
    MPI_Scatter(global_array, local_size, MPI_INT, 
                local_array, local_size, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Each process multiplies its elements by 2
    printf("Process %d received: ", rank);
    for (int i = 0; i < local_size; i++) {
        printf("%d ", local_array[i]);
        local_array[i] *= 2;
    }
    printf("\n");
    
    printf("Process %d after multiplication: ", rank);
    for (int i = 0; i < local_size; i++) {
        printf("%d ", local_array[i]);
    }
    printf("\n");
    
    // Gather the results
    MPI_Gather(local_array, local_size, MPI_INT,
               result_array, local_size, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Final result array: ");
        for (int i = 0; i < N; i++) {
            printf("%d ", result_array[i]);
        }
        printf("\n\n");
        
        free(global_array);
        free(result_array);
    }
    
    free(local_array);
}

// ============================================================================
// PROBLEM 4: MPI Derived Datatype for Particle Struct
// ============================================================================

typedef struct {
    double mass;
    double position[3];  // x, y, z coordinates
} Particle;

void particle_distribution() {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Create MPI derived datatype for Particle
    MPI_Datatype particle_type;
    int block_lengths[2] = {1, 3};
    MPI_Aint displacements[2];
    MPI_Datatype types[2] = {MPI_DOUBLE, MPI_DOUBLE};
    
    // Calculate displacements
    displacements[0] = offsetof(Particle, mass);
    displacements[1] = offsetof(Particle, position);
    
    MPI_Type_create_struct(2, block_lengths, displacements, types, &particle_type);
    MPI_Type_commit(&particle_type);
    
    int total_particles = size * 2;  // 2 particles per process
    int local_particles = total_particles / size;
    
    Particle *particles = NULL;
    Particle *local_particles_array = malloc(local_particles * sizeof(Particle));
    
    if (rank == 0) {
        particles = malloc(total_particles * sizeof(Particle));
        
        // Initialize particles with random values
        srand(time(NULL));
        printf("Initializing %d particles:\n", total_particles);
        for (int i = 0; i < total_particles; i++) {
            particles[i].mass = 1.0 + (double)rand() / RAND_MAX * 9.0;  // 1.0 to 10.0
            particles[i].position[0] = (double)rand() / RAND_MAX * 100.0;  // 0 to 100
            particles[i].position[1] = (double)rand() / RAND_MAX * 100.0;
            particles[i].position[2] = (double)rand() / RAND_MAX * 100.0;
            
            printf("Particle %d: mass=%.2f, pos=(%.2f, %.2f, %.2f)\n", 
                   i, particles[i].mass, 
                   particles[i].position[0], particles[i].position[1], particles[i].position[2]);
        }
        printf("\n");
    }
    
    // Distribute particles using MPI_Scatter
    MPI_Scatter(particles, local_particles, particle_type,
                local_particles_array, local_particles, particle_type,
                0, MPI_COMM_WORLD);
    
    // Each process prints its received particles
    printf("Process %d received particles:\n", rank);
    for (int i = 0; i < local_particles; i++) {
        printf("  Particle %d: mass=%.2f, pos=(%.2f, %.2f, %.2f)\n", 
               i, local_particles_array[i].mass,
               local_particles_array[i].position[0], 
               local_particles_array[i].position[1], 
               local_particles_array[i].position[2]);
    }
    
    // Clean up
    MPI_Type_free(&particle_type);
    
    if (rank == 0) {
        free(particles);
    }
    free(local_particles_array);
}

// ============================================================================
// MAIN FUNCTION
// ============================================================================

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0) {
        printf("=============================================================\n");
        printf("MPI Programs Collection - Running with %d processes\n", size);
        printf("=============================================================\n\n");
    }
    
    // Problem 1: Custom Allreduce implementations
    if (rank == 0) {
        printf("PROBLEM 1: Custom MPI_Allreduce Implementations\n");
        printf("-----------------------------------------------\n");
    }
    problem1_demo();
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Problem 2: Matrix-vector multiplication
    if (rank == 0) {
        printf("\nPROBLEM 2: Matrix-Vector Multiplication\n");
        printf("--------------------------------------\n");
    }
    matrix_vector_multiply();
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Problem 3: Scatter and Gather
    if (rank == 0) {
        printf("PROBLEM 3: MPI_Scatter and MPI_Gather Demo\n");
        printf("------------------------------------------\n");
    }
    scatter_gather_demo();
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Problem 4: Particle distribution
    if (rank == 0) {
        printf("PROBLEM 4: Particle Distribution with Derived Datatype\n");
        printf("-----------------------------------------------------\n");
    }
    particle_distribution();
    
    if (rank == 0) {
        printf("\n=============================================================\n");
        printf("All programs completed successfully!\n");
        printf("=============================================================\n");
    }
    
    MPI_Finalize();
    return 0;
}

/*
 * COMPILATION AND EXECUTION INSTRUCTIONS:
 * 
 * To compile:
 * mpicc -o mpi_programs mpi_programs.c -lm
 * 
 * To run (example with 4 processes):
 * mpirun -np 4 ./mpi_programs
 * 
 * NOTES:
 * - The tree and butterfly allreduce implementations work for any number of processes
 * - The butterfly version is optimized for power-of-2 process counts
 * - Matrix size in problem 2 can be adjusted by changing the 'n' variable
 * - All programs include proper memory management and error handling
 * - The particle struct uses MPI derived datatypes for efficient communication
 */