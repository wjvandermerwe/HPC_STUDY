P1 = mpi_ring_v2
P2 = mpi_ring_v4
P3 = sieve_mpi
P4 = mpi_trapezoid_1
CC = mpicc

all: $(P1) $(P2) $(P3) $(P4)

$(P1): $(P1).c
	$(CC) $(P1).c -o $(P1)

$(P2): $(P2).c
	$(CC) $(P2).c -o $(P2)

$(P3): $(P3).c
	$(CC) $(P3).c -o $(P3) -lm

$(P4): $(P4).c
	$(CC) $(P4).c -o $(P4)

clean:
	rm -vf $(P1) $(P2) $(P3) $(P4)
