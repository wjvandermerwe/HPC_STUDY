cmake_minimum_required(VERSION 3.10)
project(Codes C)

set(CMAKE_C_STANDARD 11)
set(MPI_HOME "C:/Program Files (x86)/Microsoft SDKs/MPI")  # Adjust this path

# Set include and library directories
set(MPI_INCLUDE_PATH "${MPI_HOME}/Include")
include_directories(${MPI_INCLUDE_PATH})
link_directories(${MPI_LIBRARY_PATH})

# Add each executable
add_executable(mpi_pi_mc mpi_pi_mc.c)
add_executable(mpi_ring_v2 mpi_ring_v2.c)
add_executable(mpi_ring_v4 mpi_ring_v4.c)
add_executable(mpi_trapezoid_1 mpi_trapezoid_1.c)
add_executable(sieve_mpi sieve_mpi.c)
add_executable(transpose transpose.c)

# Include MPI headers
target_include_directories(mpi_pi_mc PRIVATE ${MPI_INCLUDE_PATH})
target_include_directories(mpi_ring_v2 PRIVATE ${MPI_INCLUDE_PATH})
target_include_directories(mpi_ring_v4 PRIVATE ${MPI_INCLUDE_PATH})
target_include_directories(mpi_trapezoid_1 PRIVATE ${MPI_INCLUDE_PATH})
target_include_directories(sieve_mpi PRIVATE ${MPI_INCLUDE_PATH})
target_include_directories(transpose PRIVATE ${MPI_INCLUDE_PATH})