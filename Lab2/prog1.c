#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {

    int rank, size;
    int message;

    MPI_Init(&argc, &argv);                    // Start MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);      // Get process ID
    MPI_Comm_size(MPI_COMM_WORLD, &size);      // Get total processes

    if (size < 2) {
        if (rank == 0) {
            printf("Run with at least 2 processes!\n");
        }
        MPI_Finalize();
        return 0;
    }

    if (rank == 0) {
        message = rank;
        MPI_Send(&message, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        printf("Rank 0 sent %d to Rank 1\n", message);
    }

    if (rank == 1) {
        MPI_Recv(&message, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Rank 1 received %d from Rank 0\n", message);
    }

    MPI_Finalize();   // End MPI
    return 0;
}


