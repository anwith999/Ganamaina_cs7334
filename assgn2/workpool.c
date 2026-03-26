#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define WORK_TAG 1

int main(int argc, char *argv[]) {
    int rank, size, X;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s <seconds>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    X = atoi(argv[1]);

    double start = MPI_Wtime();
    long long local_consumed = 0;
    int send_data = rank, recv_data = 0;
    MPI_Status status;

    while (1) {
        int running = 1;

        if (rank == 0) {
            running = ((MPI_Wtime() - start) < X) ? 1 : 0;
        }

        MPI_Bcast(&running, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (!running) {
            break;
        }

        int target = (rank + 1) % size;
        int source = (rank - 1 + size) % size;

        MPI_Sendrecv(&send_data, 1, MPI_INT, target, WORK_TAG,
                     &recv_data, 1, MPI_INT, source, WORK_TAG,
                     MPI_COMM_WORLD, &status);

        local_consumed++;
    }

    long long total_consumed = 0;
    MPI_Reduce(&local_consumed, &total_consumed, 1, MPI_LONG_LONG,
               MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Total number of messages consumed: %lld\n", total_consumed);
    }

    MPI_Finalize();
    return 0;
}
