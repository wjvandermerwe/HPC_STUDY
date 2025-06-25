#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

// Function prototypes
void vector_addition(int argc, char* argv[]);
void trapezoidal_rule_v1(int argc, char* argv[]);
void trapezoidal_rule_v2(int argc, char* argv[]);
void trapezoidal_rule_v3(int argc, char* argv[]);
void matrix_vector_multiplication(int argc, char* argv[]);
void matrix_matrix_multiplication(int argc, char* argv[]);
void odd_even_sort(int argc, char* argv[]);
void parallel_quicksort(int argc, char* argv[]);
void hyperquicksort(int argc, char* argv[]);
void parallel_mergesort(int argc, char* argv[]);

// Utility functions
double f(double x);
void print_vector(double* vec, int n, char* name);
void print_matrix(double* mat, int rows, int cols, char* name);
int compare_doubles(const void* a, const void* b);
void merge(double* arr, int left, int mid, int right);
void mergesort_sequential(double* arr, int left, int right);

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if (rank == 0) {
        printf("=== MPI Programs Collection ===\n");
        printf("Choose a program to run:\n");
        printf("1. Vector Addition\n");
        printf("2. Trapezoidal Rule (Version 1)\n");
        printf("3. Trapezoidal Rule (Version 2)\n");
        printf("4. Trapezoidal Rule (Version 3)\n");
        printf("5. Matrix-Vector Multiplication\n");
        printf("6. Matrix-Matrix Multiplication\n");
        printf("7. Odd-Even Transposition Sort\n");
        printf("8. Parallel Quicksort\n");
        printf("9. Hyperquicksort\n");
        printf("10. Parallel Mergesort\n");
        printf("Enter choice (1-10): ");
    }
    
    int choice = 1; // Default to vector addition for demonstration
    if (rank == 0) {
        scanf("%d", &choice);
    }
    MPI_Bcast(&choice, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    switch(choice) {
        case 1: vector_addition(argc, argv); break;
        case 2: trapezoidal_rule_v1(argc, argv); break;
        case 3: trapezoidal_rule_v2(argc, argv); break;
        case 4: trapezoidal_rule_v3(argc, argv); break;
        case 5: matrix_vector_multiplication(argc, argv); break;
        case 6: matrix_matrix_multiplication(argc, argv); break;
        case 7: odd_even_sort(argc, argv); break;
        case 8: parallel_quicksort(argc, argv); break;
        case 9: hyperquicksort(argc, argv); break;
        case 10: parallel_mergesort(argc, argv); break;
        default: 
            if (rank == 0) printf("Invalid choice\n");
            break;
    }
    
    MPI_Finalize();
    return 0;
}

// 1. Vector Addition
void vector_addition(int argc, char* argv[]) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int n;
    double *a = NULL, *b = NULL, *c = NULL;
    double *local_a, *local_b, *local_c;
    int local_n;
    
    // Process 0 reads dimension and broadcasts
    if (rank == 0) {
        printf("\n=== Vector Addition ===\n");
        printf("Enter vector dimension: ");
        scanf("%d", &n);
        
        // Generate random vectors
        srand(time(NULL));
        a = (double*)malloc(n * sizeof(double));
        b = (double*)malloc(n * sizeof(double));
        c = (double*)malloc(n * sizeof(double));
        
        for (int i = 0; i < n; i++) {
            a[i] = (double)rand() / RAND_MAX * 10.0;
            b[i] = (double)rand() / RAND_MAX * 10.0;
        }
        
        printf("Vector A: ");
        for (int i = 0; i < (n < 10 ? n : 10); i++) {
            printf("%.2f ", a[i]);
        }
        if (n > 10) printf("...");
        printf("\n");
        
        printf("Vector B: ");
        for (int i = 0; i < (n < 10 ? n : 10); i++) {
            printf("%.2f ", b[i]);
        }
        if (n > 10) printf("...");
        printf("\n");
    }
    
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Calculate local work
    local_n = n / size;
    if (rank < n % size) local_n++;
    
    local_a = (double*)malloc(local_n * sizeof(double));
    local_b = (double*)malloc(local_n * sizeof(double));
    local_c = (double*)malloc(local_n * sizeof(double));
    
    // Scatter vectors
    MPI_Scatter(a, local_n, MPI_DOUBLE, local_a, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(b, local_n, MPI_DOUBLE, local_b, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Compute local addition
    for (int i = 0; i < local_n; i++) {
        local_c[i] = local_a[i] + local_b[i];
    }
    
    // Gather results
    MPI_Gather(local_c, n/size, MPI_DOUBLE, c, n/size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Result C = A + B: ");
        for (int i = 0; i < (n < 10 ? n : 10); i++) {
            printf("%.2f ", c[i]);
        }
        if (n > 10) printf("...");
        printf("\n");
        
        free(a); free(b); free(c);
    }
    
    free(local_a); free(local_b); free(local_c);
}

// Function for trapezoidal rule
double f(double x) {
    return x * x; // Example: f(x) = x^2
}

// 2. Trapezoidal Rule - Version 1 (Input broadcast)
void trapezoidal_rule_v1(int argc, char* argv[]) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    double a, b, h, local_sum, total_sum;
    int n, local_n;
    
    if (rank == 0) {
        printf("\n=== Trapezoidal Rule V1 ===\n");
        printf("Enter left endpoint a: ");
        scanf("%lf", &a);
        printf("Enter right endpoint b: ");
        scanf("%lf", &b);
        printf("Enter number of trapezoids n: ");
        scanf("%d", &n);
    }
    
    MPI_Bcast(&a, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&b, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    h = (b - a) / n; //global
    local_n = n / size;
    
    double local_a = a + rank * local_n * h;
    double local_b = local_a + local_n * h;
    
    local_sum = (f(local_a) + f(local_b)) / 2.0;
    for (int i = 1; i < local_n; i++) {
        double x = local_a + i * h;
        local_sum += f(x);
    }
    local_sum *= h;
    
    MPI_Reduce(&local_sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Integral approximation: %.6f\n", total_sum);
        printf("Exact value (x^2 from %.1f to %.1f): %.6f\n", a, b, (b*b*b - a*a*a)/3.0);
    }
}

// 5. Matrix-Vector Multiplication
void matrix_vector_multiplication(int argc, char* argv[]) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int m, n; // m x n matrix
    double *A = NULL, *x = NULL, *y = NULL;
    double *local_A, *local_y;
    int local_m;
    
    if (rank == 0) {
        printf("\n=== Matrix-Vector Multiplication ===\n");
        printf("Enter matrix rows m: ");
        scanf("%d", &m);
        printf("Enter matrix cols n: ");
        scanf("%d", &n);
        
        A = (double*)malloc(m * n * sizeof(double));
        x = (double*)malloc(n * sizeof(double));
        y = (double*)malloc(m * sizeof(double));
        
        // Initialize with random values
        srand(time(NULL));
        for (int i = 0; i < m * n; i++) {
            A[i] = (double)rand() / RAND_MAX * 10.0;
        }
        for (int i = 0; i < n; i++) {
            x[i] = (double)rand() / RAND_MAX * 10.0;
        }
        
        printf("Matrix A (%dx%d) and vector x generated randomly\n", m, n);
    }
    
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (x == NULL) x = (double*)malloc(n * sizeof(double));
    MPI_Bcast(x, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    local_m = m / size;
    if (rank < m % size) local_m++;
    
    local_A = (double*)malloc(local_m * n * sizeof(double));
    local_y = (double*)malloc(local_m * sizeof(double));
    
    // Scatter matrix rows
    double row_size = (m / size) * n;
    MPI_Scatter(A, row_size, MPI_DOUBLE, local_A, row_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Compute local matrix-vector multiplication
    for (int i = 0; i < local_m; i++) {
        local_y[i] = 0.0;
        for (int j = 0; j < n; j++) {
            local_y[i] += local_A[i * n + j] * x[j];
        }
    }
    
    // Gather results
    MPI_Gather(local_y, m/size, MPI_DOUBLE, y, m/size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Result y = Ax computed successfully\n");
        printf("First few elements of y: ");
        for (int i = 0; i < (m < 5 ? m : 5); i++) {
            printf("%.2f ", y[i]);
        }
        printf("\n");
        
        free(A); free(y);
    }
    
    free(x); free(local_A); free(local_y);
}

// 6. Matrix-Matrix Multiplication
void matrix_matrix_multiplication(int argc, char* argv[]) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int m, n, p; // A is mxn, B is nxp, C is mxp
    double *A = NULL, *B = NULL, *C = NULL;
    double *local_A, *local_C;
    int local_m;
    
    if (rank == 0) {
        printf("\n=== Matrix-Matrix Multiplication ===\n");
        printf("Enter dimensions m, n, p (A:mxn, B:nxp): ");
        scanf("%d %d %d", &m, &n, &p);
        
        A = (double*)malloc(m * n * sizeof(double));
        B = (double*)malloc(n * p * sizeof(double));
        C = (double*)malloc(m * p * sizeof(double));
        
        // Initialize with random values
        srand(time(NULL));
        for (int i = 0; i < m * n; i++) {
            A[i] = (double)rand() / RAND_MAX * 5.0;
        }
        for (int i = 0; i < n * p; i++) {
            B[i] = (double)rand() / RAND_MAX * 5.0;
        }
        
        printf("Matrices A(%dx%d) and B(%dx%d) initialized\n", m, n, n, p);
    }
    
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&p, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (B == NULL) B = (double*)malloc(n * p * sizeof(double));
    MPI_Bcast(B, n * p, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    local_m = m / size;
    if (rank < m % size) local_m++;
    
    local_A = (double*)malloc(local_m * n * sizeof(double));
    local_C = (double*)malloc(local_m * p * sizeof(double));
    
    // Scatter rows of A
    MPI_Scatter(A, (m/size) * n, MPI_DOUBLE, local_A, (m/size) * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Compute local matrix multiplication
    for (int i = 0; i < local_m; i++) {
        for (int j = 0; j < p; j++) {
            local_C[i * p + j] = 0.0;
            for (int k = 0; k < n; k++) {
                local_C[i * p + j] += local_A[i * n + k] * B[k * p + j];
            }
        }
    }
    
    // Gather results
    MPI_Gather(local_C, (m/size) * p, MPI_DOUBLE, C, (m/size) * p, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Matrix multiplication C = A * B completed\n");
        printf("Result matrix C is %dx%d\n", m, p);
        
        free(A); free(B); free(C);
    }
    
    free(local_A); free(local_C);
}

// 7. Odd-Even Transposition Sort
void odd_even_sort(int argc, char* argv[]) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int n, local_n;
    double *data = NULL, *local_data;
    
    if (rank == 0) {
        printf("\n=== Odd-Even Transposition Sort ===\n");
        printf("Enter array size: ");
        scanf("%d", &n);
        
        data = (double*)malloc(n * sizeof(double));
        srand(time(NULL));
        for (int i = 0; i < n; i++) {
            data[i] = (double)rand() / RAND_MAX * 100.0;
        }
        
        printf("Original array (first 10): ");
        for (int i = 0; i < (n < 10 ? n : 10); i++) {
            printf("%.1f ", data[i]);
        }
        printf("\n");
    }
    
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    local_n = n / size;
    local_data = (double*)malloc(local_n * sizeof(double));
    
    MPI_Scatter(data, local_n, MPI_DOUBLE, local_data, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Sort local data
    qsort(local_data, local_n, sizeof(double), compare_doubles);
    
    // Odd-even transposition
    for (int phase = 0; phase < size; phase++) {
        if (phase % 2 == 0) { // Even phase
            if (rank % 2 == 0 && rank < size - 1) {
                // Send to right, receive from right
                double *temp = (double*)malloc(local_n * sizeof(double));
                MPI_Sendrecv(local_data, local_n, MPI_DOUBLE, rank + 1, 0,
                           temp, local_n, MPI_DOUBLE, rank + 1, 0,
                           MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                // Merge and keep smaller elements
                double *merged = (double*)malloc(2 * local_n * sizeof(double));
                int i = 0, j = 0, k = 0;
                while (i < local_n && j < local_n) {
                    if (local_data[i] <= temp[j]) {
                        merged[k++] = local_data[i++];
                    } else {
                        merged[k++] = temp[j++];
                    }
                }
                
                for (int i = 0; i < local_n; i++) {
                    local_data[i] = merged[i];
                }
                
                free(temp); free(merged);
            } else if (rank % 2 == 1) {
                // Send to left, receive from left
                double *temp = (double*)malloc(local_n * sizeof(double));
                MPI_Sendrecv(local_data, local_n, MPI_DOUBLE, rank - 1, 0,
                           temp, local_n, MPI_DOUBLE, rank - 1, 0,
                           MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                // Merge and keep larger elements
                double *merged = (double*)malloc(2 * local_n * sizeof(double));
                int i = 0, j = 0, k = 0;
                while (i < local_n && j < local_n) {
                    if (local_data[i] <= temp[j]) {
                        merged[k++] = local_data[i++];
                    } else {
                        merged[k++] = temp[j++];
                    }
                }
                
                for (int i = 0; i < local_n; i++) {
                    local_data[i] = merged[local_n + i];
                }
                
                free(temp); free(merged);
            }
        } else { // Odd phase
            if (rank % 2 == 1 && rank < size - 1) {
                // Similar logic for odd phase
                double *temp = (double*)malloc(local_n * sizeof(double));
                MPI_Sendrecv(local_data, local_n, MPI_DOUBLE, rank + 1, 0,
                           temp, local_n, MPI_DOUBLE, rank + 1, 0,
                           MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                double *merged = (double*)malloc(2 * local_n * sizeof(double));
                int i = 0, j = 0, k = 0;
                while (i < local_n && j < local_n) {
                    if (local_data[i] <= temp[j]) {
                        merged[k++] = local_data[i++];
                    } else {
                        merged[k++] = temp[j++];
                    }
                }
                
                for (int i = 0; i < local_n; i++) {
                    local_data[i] = merged[i];
                }
                
                free(temp); free(merged);
            } else if (rank % 2 == 0 && rank > 0) {
                double *temp = (double*)malloc(local_n * sizeof(double));
                MPI_Sendrecv(local_data, local_n, MPI_DOUBLE, rank - 1, 0,
                           temp, local_n, MPI_DOUBLE, rank - 1, 0,
                           MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                double *merged = (double*)malloc(2 * local_n * sizeof(double));
                int i = 0, j = 0, k = 0;
                while (i < local_n && j < local_n) {
                    if (local_data[i] <= temp[j]) {
                        merged[k++] = local_data[i++];
                    } else {
                        merged[k++] = temp[j++];
                    }
                }

                for (int i = 0; i < local_n; i++) {
                    local_data[i] = merged[local_n + i];
                }
                
                free(temp); free(merged);
            }
        }
    }
    
    MPI_Gather(local_data, local_n, MPI_DOUBLE, data, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Sorted array (first 10): ");
        for (int i = 0; i < (n < 10 ? n : 10); i++) {
            printf("%.1f ", data[i]);
        }
        printf("\n");
        free(data);
    }
    
    free(local_data);
}

// 8. Parallel Quicksort
void parallel_quicksort(int argc, char* argv[]) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int n, local_n;
    double *data = NULL, *local_data;
    
    if (rank == 0) {
        printf("\n=== Parallel Quicksort ===\n");
        printf("Enter array size: ");
        scanf("%d", &n);
        
        data = (double*)malloc(n * sizeof(double));
        srand(time(NULL));
        for (int i = 0; i < n; i++) {
            data[i] = (double)rand() / RAND_MAX * 100.0;
        }
        
        printf("Original array (first 10): ");
        for (int i = 0; i < (n < 10 ? n : 10); i++) {
            printf("%.1f ", data[i]);
        }
        printf("\n");
    }
    
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    local_n = n / size;
    local_data = (double*)malloc(local_n * sizeof(double));
    
    MPI_Scatter(data, local_n, MPI_DOUBLE, local_data, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Sort local data using quicksort
    qsort(local_data, local_n, sizeof(double), compare_doubles);
    
    // Implement parallel quicksort phases
    for (int step = 1; step < size; step *= 2) {
        double pivot;
        
        // Choose pivot (process 0 in each group chooses)
        if (rank % (2 * step) == 0) {
            pivot = local_data[local_n / 2]; // Choose middle element as pivot
        }
        
        // Broadcast pivot within each group
        int group_root = (rank / (2 * step)) * (2 * step);
        MPI_Bcast(&pivot, 1, MPI_DOUBLE, group_root, MPI_COMM_WORLD);
        
        // Partition local data
        int left_count = 0;
        for (int i = 0; i < local_n; i++) {
            if (local_data[i] < pivot) left_count++;
        }
        
        double *left_data = (double*)malloc(left_count * sizeof(double));
        double *right_data = (double*)malloc((local_n - left_count) * sizeof(double));
        
        int left_idx = 0, right_idx = 0;
        for (int i = 0; i < local_n; i++) {
            if (local_data[i] < pivot) {
                left_data[left_idx++] = local_data[i];
            } else {
                right_data[right_idx++] = local_data[i];
            }
        }
        
        // Exchange data with partner
        int partner = rank ^ step;
        if (partner < size) {
            int send_count, recv_count;
            double *send_data, *recv_data;
            
            if (rank < partner) {
                // Send right, receive left
                send_count = local_n - left_count;
                send_data = right_data;
                MPI_Sendrecv(&send_count, 1, MPI_INT, partner, 0,
                           &recv_count, 1, MPI_INT, partner, 0,
                           MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                recv_data = (double*)malloc(recv_count * sizeof(double));
                MPI_Sendrecv(send_data, send_count, MPI_DOUBLE, partner, 1,
                           recv_data, recv_count, MPI_DOUBLE, partner, 1,
                           MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                // Merge left data with received data
                local_n = left_count + recv_count;
                free(local_data);
                local_data = (double*)malloc(local_n * sizeof(double));
                
                int i = 0, j = 0, k = 0;
                while (i < left_count && j < recv_count) {
                    if (left_data[i] <= recv_data[j]) {
                        local_data[k++] = left_data[i++];
                    } else {
                        local_data[k++] = recv_data[j++];
                    }
                }
                while (i < left_count) local_data[k++] = left_data[i++];
                while (j < recv_count) local_data[k++] = recv_data[j++];
                
            } else {
                // Send left, receive right
                send_count = left_count;
                send_data = left_data;
                MPI_Sendrecv(&send_count, 1, MPI_INT, partner, 0,
                           &recv_count, 1, MPI_INT, partner, 0,
                           MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                recv_data = (double*)malloc(recv_count * sizeof(double));
                MPI_Sendrecv(send_data, send_count, MPI_DOUBLE, partner, 1,
                           recv_data, recv_count, MPI_DOUBLE, partner, 1,
                           MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                // Merge right data with received data
                local_n = (local_n - left_count) + recv_count;
                free(local_data);
                local_data = (double*)malloc(local_n * sizeof(double));
                
                int i = 0, j = 0, k = 0;
                while (i < recv_count && j < (local_n - recv_count)) {
                    if (recv_data[i] <= right_data[j]) {
                        local_data[k++] = recv_data[i++];
                    } else {
                        local_data[k++] = right_data[j++];
                    }
                }
                while (i < recv_count) local_data[k++] = recv_data[i++];
                while (j < (local_n - recv_count)) local_data[k++] = right_data[j++];
            }
            
            free(recv_data);
        }
        
        free(left_data);
        free(right_data);
    }
    
    // Gather all local sizes first
    int *all_local_n = NULL;
    if (rank == 0) {
        all_local_n = (int*)malloc(size * sizeof(int));
    }
    MPI_Gather(&local_n, 1, MPI_INT, all_local_n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Calculate displacements
    int *displs = NULL;
    if (rank == 0) {
        displs = (int*)malloc(size * sizeof(int));
        displs[0] = 0;
        for (int i = 1; i < size; i++) {
            displs[i] = displs[i-1] + all_local_n[i-1];
        }
        free(data);
        data = (double*)malloc(displs[size-1] + all_local_n[size-1] * sizeof(double));
    }
    
    // Gather sorted data
    MPI_Gatherv(local_data, local_n, MPI_DOUBLE, data, all_local_n, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Sorted array (first 10): ");
        int total_n = displs[size-1] + all_local_n[size-1];
        for (int i = 0; i < (total_n < 10 ? total_n : 10); i++) {
            printf("%.1f ", data[i]);
        }
        printf("\n");
        free(data);
        free(all_local_n);
        free(displs);
    }
    
    free(local_data);
}

// 9. Hyperquicksort
void hyperquicksort(int argc, char* argv[]) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int n, local_n;
    double *data = NULL, *local_data;
    
    if (rank == 0) {
        printf("\n=== Hyperquicksort ===\n");
        printf("Enter array size: ");
        scanf("%d", &n);
        
        data = (double*)malloc(n * sizeof(double));
        srand(time(NULL));
        for (int i = 0; i < n; i++) {
            data[i] = (double)rand() / RAND_MAX * 100.0;
        }
        
        printf("Original array (first 10): ");
        for (int i = 0; i < (n < 10 ? n : 10); i++) {
            printf("%.1f ", data[i]);
        }
        printf("\n");
    }
    
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    local_n = n / size;
    local_data = (double*)malloc(local_n * sizeof(double));
    
    MPI_Scatter(data, local_n, MPI_DOUBLE, local_data, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Sort local data
    qsort(local_data, local_n, sizeof(double), compare_doubles);
    
    // Hyperquicksort algorithm - recursive divide and conquer
    int active_procs = size;
    int proc_group_size = size;
    int my_group_rank = rank;
    
    while (proc_group_size > 1) {
        // Select pivot (median of medians approach)
        double local_median = local_data[local_n / 2];
        double *all_medians = NULL;
        
        if (my_group_rank == 0) {
            all_medians = (double*)malloc(proc_group_size * sizeof(double));
        }
        
        // Create communicator for current group
        MPI_Comm group_comm;
        int group_start = (rank / proc_group_size) * proc_group_size;
        int color = rank / proc_group_size;
        MPI_Comm_split(MPI_COMM_WORLD, color, rank, &group_comm);
        
        MPI_Gather(&local_median, 1, MPI_DOUBLE, all_medians, 1, MPI_DOUBLE, 0, group_comm);
        
        double pivot;
        if (my_group_rank == 0) {
            qsort(all_medians, proc_group_size, sizeof(double), compare_doubles);
            pivot = all_medians[proc_group_size / 2];
            free(all_medians);
        }
        
        MPI_Bcast(&pivot, 1, MPI_DOUBLE, 0, group_comm);
        
        // Partition local data around pivot
        int left_count = 0;
        for (int i = 0; i < local_n; i++) {
            if (local_data[i] < pivot) left_count++;
        }
        
        double *left_data = (double*)malloc(left_count * sizeof(double));
        double *right_data = (double*)malloc((local_n - left_count) * sizeof(double));
        
        int left_idx = 0, right_idx = 0;
        for (int i = 0; i < local_n; i++) {
            if (local_data[i] < pivot) {
                left_data[left_idx++] = local_data[i];
            } else {
                right_data[right_idx++] = local_data[i];
            }
        }
        
        
        // Determine which half of processors to work with
        int half_size = proc_group_size / 2;
        if (my_group_rank < half_size) {
            // Work with left half - exchange right data
            int partner = my_group_rank + half_size + group_start;
            if (partner < group_start + proc_group_size) {
                int send_count = local_n - left_count;
                int recv_count;
                
                MPI_Sendrecv(&send_count, 1, MPI_INT, partner, 0,
                           &recv_count, 1, MPI_INT, partner, 0,
                           MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                double *recv_data = (double*)malloc(recv_count * sizeof(double));
                MPI_Sendrecv(right_data, send_count, MPI_DOUBLE, partner, 1,
                           recv_data, recv_count, MPI_DOUBLE, partner, 1,
                           MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                // Merge left data with received data
                local_n = left_count + recv_count;
                free(local_data);
                local_data = (double*)malloc(local_n * sizeof(double));
                
                merge_sorted_arrays(left_data, left_count, recv_data, recv_count, local_data);
                
                free(recv_data);
            } else {
                // No partner, keep left data only
                local_n = left_count;
                free(local_data);
                local_data = (double*)malloc(local_n * sizeof(double));
                memcpy(local_data, left_data, local_n * sizeof(double));
            }
        } else {
            // Work with right half - exchange left data
            int partner = my_group_rank - half_size + group_start;
            int send_count = left_count;
            int recv_count;
            
            MPI_Sendrecv(&send_count, 1, MPI_INT, partner, 0,
                       &recv_count, 1, MPI_INT, partner, 0,
                       MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            double *recv_data = (double*)malloc(recv_count * sizeof(double));
            MPI_Sendrecv(left_data, send_count, MPI_DOUBLE, partner, 1,
                       recv_data, recv_count, MPI_DOUBLE, partner, 1,
                       MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            // Merge right data with received data
            local_n = (local_n - left_count) + recv_count;
            free(local_data);
            local_data = (double*)malloc(local_n * sizeof(double));
            
            merge_sorted_arrays(recv_data, recv_count, right_data, local_n - recv_count, local_data);
            
            free(recv_data);
            my_group_rank -= half_size;
        }
        
        free(left_data);
        free(right_data);
        
        proc_group_size = half_size;
        MPI_Comm_free(&group_comm);
    }
    
    // Gather results with variable sizes
    int *all_local_n = NULL;
    int *displs = NULL;
    if (rank == 0) {
        all_local_n = (int*)malloc(size * sizeof(int));
        displs = (int*)malloc(size * sizeof(int));
    }
    
    MPI_Gather(&local_n, 1, MPI_INT, all_local_n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < size; i++) {
            displs[i] = displs[i-1] + all_local_n[i-1];
        }
        int total_n = displs[size-1] + all_local_n[size-1];
        free(data);
        data = (double*)malloc(total_n * sizeof(double));
    }
    
    MPI_Gatherv(local_data, local_n, MPI_DOUBLE, data, all_local_n, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Sorted array (first 10): ");
        int total_n = displs[size-1] + all_local_n[size-1];
        for (int i = 0; i < (total_n < 10 ? total_n : 10); i++) {
            printf("%.1f ", data[i]);
        }
        printf("\n");
        free(data);
        free(all_local_n);
        free(displs);
    }
    
    free(local_data);
}

// 10. Parallel Mergesort
void parallel_mergesort(int argc, char* argv[]) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int n, local_n;
    double *data = NULL, *local_data;
    
    if (rank == 0) {
        printf("\n=== Parallel Mergesort ===\n");
        printf("Enter array size: ");
        scanf("%d", &n);
        
        data = (double*)malloc(n * sizeof(double));
        srand(time(NULL));
        for (int i = 0; i < n; i++) {
            data[i] = (double)rand() / RAND_MAX * 100.0;
        }
        
        printf("Original array (first 10): ");
        for (int i = 0; i < (n < 10 ? n : 10); i++) {
            printf("%.1f ", data[i]);
        }
        printf("\n");
    }
    
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    local_n = n / size;
    local_data = (double*)malloc(local_n * sizeof(double));
    
    MPI_Scatter(data, local_n, MPI_DOUBLE, local_data, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Sort local data using sequential mergesort
    mergesort_sequential(local_data, 0, local_n - 1);
    
    // Parallel merge phase
    for (int step = 1; step < size; step *= 2) {
        if (rank % (2 * step) == 0) {
            if (rank + step < size) {
                // Receive data from partner
                int recv_count;
                MPI_Recv(&recv_count, 1, MPI_INT, rank + step, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                double *recv_data = (double*)malloc(recv_count * sizeof(double));
                MPI_Recv(recv_data, recv_count, MPI_DOUBLE, rank + step, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                // Merge local_data and recv_data
                double *merged = (double*)malloc((local_n + recv_count) * sizeof(double));
                
                int i = 0, j = 0, k = 0;
                while (i < local_n && j < recv_count) {
                    if (local_data[i] <= recv_data[j]) {
                        merged[k++] = local_data[i++];
                    } else {
                        merged[k++] = recv_data[j++];
                    }
                }
                while (i < local_n) merged[k++] = local_data[i++];
                while (j < recv_count) merged[k++] = recv_data[j++];
                
                free(local_data);
                free(recv_data);
                local_data = merged;
                local_n += recv_count;
            }
        } else if (rank % (2 * step) == step) {
            // Send data to partner
            int partner = rank - step;
            MPI_Send(&local_n, 1, MPI_INT, partner, 0, MPI_COMM_WORLD);
            MPI_Send(local_data, local_n, MPI_DOUBLE, partner, 1, MPI_COMM_WORLD);
            break; // This process is done
        }
    }
    
    if (rank == 0) {
        printf("Sorted array (first 10): ");
        for (int i = 0; i < (local_n < 10 ? local_n : 10); i++) {
            printf("%.1f ", local_data[i]);
        }
        printf("\n");
        free(data);
    }
    
    free(local_data);
}

// Utility functions
int compare_doubles(const void* a, const void* b) {
    double da = *(const double*)a;
    double db = *(const double*)b;
    return (da > db) - (da < db);
}

void merge(double* arr, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;
    
    double* L = (double*)malloc(n1 * sizeof(double));
    double* R = (double*)malloc(n2 * sizeof(double));
    
    for (int i = 0; i < n1; i++)
        L[i] = arr[left + i];
    for (int j = 0; j < n2; j++)
        R[j] = arr[mid + 1 + j];
    
    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }
    
    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }
    
    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
    
    free(L);
    free(R);
}

void mergesort_sequential(double* arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        
        mergesort_sequential(arr, left, mid);
        mergesort_sequential(arr, mid + 1, right);
        
        merge(arr, left, mid, right);
    }
}

void merge_sorted_arrays(double* arr1, int n1, double* arr2, int n2, double* result) {
    int i = 0, j = 0, k = 0;
    while (i < n1 && j < n2) {
        if (arr1[i] <= arr2[j]) {
            result[k++] = arr1[i++];
        } else {
            result[k++] = arr2[j++];
        }
    }
    while (i < n1) result[k++] = arr1[i++];
    while (j < n2) result[k++] = arr2[j++];
}

void print_vector(double* vec, int n, char* name) {
    printf("%s: ", name);
    for (int i = 0; i < (n < 10 ? n : 10); i++) {
        printf("%.2f ", vec[i]);
    }
    if (n > 10) printf("...");
    printf("\n");
}

void print_matrix(double* mat, int rows, int cols, char* name) {
    printf("%s (%dx%d):\n", name, rows, cols);
    for (int i = 0; i < (rows < 5 ? rows : 5); i++) {
        for (int j = 0; j < (cols < 5 ? cols : 5); j++) {
            printf("%.2f ", mat[i * cols + j]);
        }
        if (cols > 5) printf("...");
        printf("\n");
    }
    if (rows > 5) printf("...\n");
}