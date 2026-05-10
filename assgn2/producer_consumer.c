#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define WORK_TAG 1
#define REQUEST_TAG 2
#define ACK_TAG 3
#define ABORT_TAG 4

int main(int argc, char *argv[]) {
    int rank, size;
    int X;

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
    srand(time(NULL) + rank);

    int num_producers = (size - 1) / 2;
    int broker = 0;
    double start_time = MPI_Wtime();

    if (rank == broker) {
        int buffer_size = 2 * num_producers;
        int *buffer = (int *)malloc(buffer_size * sizeof(int));
        int count = 0;
        int total_consumed = 0;
        MPI_Status status;

        while (MPI_Wtime() - start_time < X) {
            int flag = 0;
            MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);

            if (!flag) {
                continue;
            }

            if (status.MPI_TAG == WORK_TAG) {
                int data;
                MPI_Recv(&data, 1, MPI_INT, status.MPI_SOURCE, WORK_TAG, MPI_COMM_WORLD, &status);

                if (count < buffer_size) {
                    buffer[count++] = data;
                }

                MPI_Send(NULL, 0, MPI_INT, status.MPI_SOURCE, ACK_TAG, MPI_COMM_WORLD);
            }
            else if (status.MPI_TAG == REQUEST_TAG) {
                MPI_Recv(NULL, 0, MPI_INT, status.MPI_SOURCE, REQUEST_TAG, MPI_COMM_WORLD, &status);

                if (count > 0) {
                    int data = buffer[--count];
                    MPI_Send(&data, 1, MPI_INT, status.MPI_SOURCE, WORK_TAG, MPI_COMM_WORLD);
                    total_consumed++;
                } else {
                    MPI_Send(NULL, 0, MPI_INT, status.MPI_SOURCE, ACK_TAG, MPI_COMM_WORLD);
                }
            }
        }

        for (int i = 1; i < size; i++) {
            MPI_Send(NULL, 0, MPI_INT, i, ABORT_TAG, MPI_COMM_WORLD);
        }

        int grand_total = total_consumed;

        for (int i = num_producers + 1; i < size; i++) {
            int local_count = 0;
            MPI_Recv(&local_count, 1, MPI_INT, i, ACK_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            grand_total += local_count;
        }

        printf("Total number of messages consumed: %d\n", grand_total);
        free(buffer);
    }
    else if (rank >= 1 && rank <= num_producers) {
        MPI_Status status;

        while (MPI_Wtime() - start_time < X) {
            int data = rand();

            MPI_Send(&data, 1, MPI_INT, broker, WORK_TAG, MPI_COMM_WORLD);
            MPI_Recv(NULL, 0, MPI_INT, broker, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (status.MPI_TAG == ABORT_TAG) {
                break;
            }
        }

        int flag = 0;
        MPI_Iprobe(broker, ABORT_TAG, MPI_COMM_WORLD, &flag, &status);
        if (flag) {
            MPI_Recv(NULL, 0, MPI_INT, broker, ABORT_TAG, MPI_COMM_WORLD, &status);
        }
    }
    else {
        int local_count = 0;
        MPI_Status status;

        while (MPI_Wtime() - start_time < X) {
            MPI_Send(NULL, 0, MPI_INT, broker, REQUEST_TAG, MPI_COMM_WORLD);

            int data;
            MPI_Recv(&data, 1, MPI_INT, broker, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (status.MPI_TAG == WORK_TAG) {
                local_count++;
            } else if (status.MPI_TAG == ABORT_TAG) {
                break;
            }
        }

        int flag = 0;
        MPI_Iprobe(broker, ABORT_TAG, MPI_COMM_WORLD, &flag, &status);
        if (flag) {
            MPI_Recv(NULL, 0, MPI_INT, broker, ABORT_TAG, MPI_COMM_WORLD, &status);
        }

        MPI_Send(&local_count, 1, MPI_INT, broker, ACK_TAG, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
