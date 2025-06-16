/*
 * Monte Carlo Computation of π using MPI
 * 
 * Algorithm:
 * - Generate random points (x,y) in square [-1,1] x [-1,1]
 * - Count points inside unit circle (x² + y² ≤ 1)
 * - Ratio of points inside circle to total points approximates π/4
 * - Therefore π ≈ 4 * (points_inside / total_points)
 *
 * Architecture:
 * - Server process (last rank) generates random numbers
 * - Worker processes request chunks, compute results, use MPI_Allreduce
 * - Continue until error threshold met or max iterations reached
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <time.h>

#define CHUNK_SIZE 1000      // Number of random pairs per chunk
#define MAX_ITERATIONS 1000  // Maximum number of iterations
#define ERROR_THRESHOLD 0.001 // Stop when |π_estimate - π| < threshold
#define REQUEST_TAG 1        // Tag for requesting random numbers
#define DATA_TAG 2          // Tag for sending random data
#define STOP_TAG 3          // Tag to signal workers to stop

// Structure to send random number pairs
typedef struct {
    double x;
    double y;
} Point;

// Server process function
void server_process(MPI_Comm world, int server_rank, int num_workers) {
    printf("Server process %d started, serving %d workers\n", server_rank, num_workers);
    
    // Initialize random seed
    srand(time(NULL) + server_rank);
    
    Point *chunk = malloc(CHUNK_SIZE * sizeof(Point));
    int active_workers = num_workers;
    int total_chunks_sent = 0;
    
    while (active_workers > 0) {
        MPI_Status status;
        int request;
        
        // Wait for request from any worker
        MPI_Recv(&request, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, world, &status);
        
        if (status.MPI_TAG == REQUEST_TAG) {
            // Generate chunk of random points
            for (int i = 0; i < CHUNK_SIZE; i++) {
                // Generate random points in [-1, 1] x [-1, 1]
                chunk[i].x = 2.0 * ((double)rand() / RAND_MAX) - 1.0;
                chunk[i].y = 2.0 * ((double)rand() / RAND_MAX) - 1.0;
            }
            
            // Send chunk to requesting worker
            MPI_Send(chunk, CHUNK_SIZE * 2, MPI_DOUBLE, status.MPI_SOURCE, DATA_TAG, world);
            total_chunks_sent++;
            
            printf("Server: Sent chunk %d to worker %d\n", total_chunks_sent, status.MPI_SOURCE);
            
        } else if (status.MPI_TAG == STOP_TAG) {
            // Worker is done
            active_workers--;
            printf("Server: Worker %d finished. %d workers remaining.\n", 
                   status.MPI_SOURCE, active_workers);
        }
    }
    
    printf("Server: All workers finished. Total chunks sent: %d\n", total_chunks_sent);
    free(chunk);
}

// Worker process function
void worker_process(MPI_Comm world, MPI_Comm workers, int world_rank, int server_rank) {
    int worker_rank;
    int num_workers;
    MPI_Comm_rank(workers, &worker_rank);
    MPI_Comm_size(workers, &num_workers);
    
    printf("Worker %d (world rank %d) started\n", worker_rank, world_rank);
    
    Point *chunk = malloc(CHUNK_SIZE * sizeof(Point));
    int iteration = 0;
    int total_points = 0;
    int total_inside = 0;
    double pi_estimate = 0.0;
    double error = 1.0;
    
    while (iteration < MAX_ITERATIONS && error > ERROR_THRESHOLD) {
        // Request chunk from server
        int request = 1;
        MPI_Send(&request, 1, MPI_INT, server_rank, REQUEST_TAG, world);
        
        // Receive chunk from server
        MPI_Status status;
        MPI_Recv(chunk, CHUNK_SIZE * 2, MPI_DOUBLE, server_rank, DATA_TAG, world, &status);
        
        // Process chunk - count points inside unit circle
        int local_inside = 0;
        for (int i = 0; i < CHUNK_SIZE; i++) {
            double distance_squared = chunk[i].x * chunk[i].x + chunk[i].y * chunk[i].y;
            if (distance_squared <= 1.0) {
                local_inside++;
            }
        }
        
        // Update local counters
        total_inside += local_inside;
        total_points += CHUNK_SIZE;
        
        // Collective communication among workers to get global estimate
        int global_inside, global_points;
        MPI_Allreduce(&total_inside, &global_inside, 1, MPI_INT, MPI_SUM, workers);
        MPI_Allreduce(&total_points, &global_points, 1, MPI_INT, MPI_SUM, workers);
        
        // Calculate π estimate
        pi_estimate = 4.0 * (double)global_inside / (double)global_points;
        error = fabs(pi_estimate - M_PI);
        
        iteration++;
        
        // Worker 0 prints progress
        if (worker_rank == 0) {
            printf("Iteration %d: π estimate = %.6f, error = %.6f, total points = %d\n",
                   iteration, pi_estimate, error, global_points);
        }
        
        // Check stopping criteria
        if (error <= ERROR_THRESHOLD) {
            if (worker_rank == 0) {
                printf("Convergence achieved! π estimate = %.6f (error = %.6f)\n", 
                       pi_estimate, error);
            }
            break;
        }
        
        if (iteration >= MAX_ITERATIONS) {
            if (worker_rank == 0) {
                printf("Maximum iterations reached. π estimate = %.6f (error = %.6f)\n", 
                       pi_estimate, error);
            }
            break;
        }
    }
    
    // Notify server that this worker is done
    int stop_signal = 0;
    MPI_Send(&stop_signal, 1, MPI_INT, server_rank, STOP_TAG, world);
    
    printf("Worker %d finished: local inside = %d, local total = %d\n", 
           worker_rank, total_inside, total_points);
    
    free(chunk);
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    
    MPI_Comm world = MPI_COMM_WORLD, workers;
    MPI_Group world_group, worker_group;
    
    int numprocs, myid, server;
    MPI_Comm_size(world, &numprocs);
    MPI_Comm_rank(world, &myid);
    
    if (numprocs < 2) {
        if (myid == 0) {
            printf("Error: Need at least 2 processes (1 server + 1 worker)\n");
        }
        MPI_Finalize();
        return 1;
    } // if processes are shutting down stop
    
    // Last process is the server
    server = numprocs - 1;
    int num_workers = numprocs - 1;
    
    // Create worker communicator (exclude server)
    MPI_Comm_group(world, &world_group);
    int ranks[1] = {server};
    MPI_Group_excl(world_group, 1, ranks, &worker_group);
    MPI_Comm_create(world, worker_group, &workers);
    
    if (myid == 0) {
        printf("=======================================================\n");
        printf("Monte Carlo π Computation using MPI\n");
        printf("Total processes: %d (1 server + %d workers)\n", numprocs, num_workers);
        printf("Chunk size: %d points per request\n", CHUNK_SIZE);
        printf("Error threshold: %.6f\n", ERROR_THRESHOLD);
        printf("Maximum iterations: %d\n", MAX_ITERATIONS);
        printf("Actual π = %.10f\n", M_PI);
        printf("=======================================================\n\n");
    }
    
    double start_time = MPI_Wtime();
    
    if (myid == server) {
        // Server process
        server_process(world, server, num_workers);
    } else {
        // Worker process
        worker_process(world, workers, myid, server);
    }
    
    // Synchronize all processes
    MPI_Barrier(world);
    
    double end_time = MPI_Wtime();
    
    if (myid == 0) {
        printf("\n=======================================================\n");
        printf("Monte Carlo π computation completed!\n");
        printf("Total execution time: %.3f seconds\n", end_time - start_time);
        printf("=======================================================\n");
    }
    
    // Clean up
    if (myid != server) {
        MPI_Comm_free(&workers);
    }
    MPI_Group_free(&worker_group);
    MPI_Group_free(&world_group);
    
    MPI_Finalize();
    return 0;
}

/*
 * COMPILATION AND EXECUTION:
 * 
 * Compile:
 * mpicc -o monte_carlo_pi monte_carlo_pi.c -lm
 * 
 * Run with 4 processes (3 workers + 1 server):
 * mpirun -np 4 ./monte_carlo_pi
 * 
 * Run with 8 processes (7 workers + 1 server):
 * mpirun -np 8 ./monte_carlo_pi
 * 
 * ALGORITHM EXPLANATION:
 * 
 * 1. Monte Carlo Method:
 *    - Generate random points (x,y) in square [-1,1] × [-1,1]
 *    - Count points inside unit circle: x² + y² ≤ 1
 *    - Ratio = (points inside circle) / (total points) ≈ π/4
 *    - Therefore: π ≈ 4 × ratio
 * 
 * 2. MPI Architecture:
 *    - Server process generates random numbers on demand
 *    - Worker processes request chunks, compute local results
 *    - MPI_Allreduce combines results from all workers
 *    - Continue until convergence or max iterations
 * 
 * 3. Communication Pattern:
 *    - Workers → Server: REQUEST_TAG (request for random numbers)
 *    - Server → Workers: DATA_TAG (chunk of random points)
 *    - Workers → Server: STOP_TAG (worker finished)
 *    - Workers ↔ Workers: MPI_Allreduce (combine results)
 * 
 * PERFORMANCE NOTES:
 * - Adjust CHUNK_SIZE for optimal performance
 * - Larger chunks reduce communication overhead
 * - Smaller chunks provide better load balancing
 * - Error threshold determines accuracy vs. computation time
 */