mpiexec -n 4 ./mpi_ring_v2 |& tee -a ring2.out
echo ---------ring2 is done-------------

mpiexec -n 4 ./mpi_ring_v4 |& tee -a ring4.out
echo ---------ring4 is done-------------

mpiexec -n 4 ./sieve_mpi 3000 |& tee -a sieve_mpi.out
echo ---------sieve_mpi is done-------------

mpiexec -n 4 ./mpi_trapezoid_1 0.0 1.0 1000000 |& tee mpi_trape.out
echo ---------mpi_trapezoid is done-------------
