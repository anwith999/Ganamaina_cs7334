#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int value;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) printf("Run with at least 2 processes.\n");
        MPI_Finalize();
        return 0;
    }

    int next = (rank + 1) % size;
    int prev = (rank - 1 + size) % size;

    if (rank == 0) {
        // create random number
        srand((unsigned)time(NULL));
        value = rand() % 1000;

        printf("Rank 0 starting value = %d\n", value);

        // send to rank 1 (next)
        MPI_Send(&value, 1, MPI_INT, next, 0, MPI_COMM_WORLD);

        // receive back from last rank (prev)
        MPI_Recv(&value, 1, MPI_INT, prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        printf("Rank 0 received back value = %d\n", value);
    } else {
        // receive from previous rank
        MPI_Recv(&value, 1, MPI_INT, prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Rank %d received %d from Rank %d\n", rank, value, prev);

        // forward to next rank
        MPI_Send(&value, 1, MPI_INT, next, 0, MPI_COMM_WORLD);
        printf("Rank %d sent %d to Rank %d\n", rank, value, next);
    }

    MPI_Finalize();
    return 0;
}


