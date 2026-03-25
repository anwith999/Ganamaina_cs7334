/* noncontiguous access with a single collective I/O function */
#include "mpi.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define FILESIZE 1024
#define INTS_PER_BLK 1

int main(int argc, char **argv)
{
    int *buf, rank, nprocs, nints, bufsize;
    MPI_File fh;
    MPI_Datatype filetype;

    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get rank
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Get number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    bufsize = FILESIZE / nprocs;
    buf = (int *) malloc(bufsize);
    nints = bufsize / sizeof(int);

    // Fill buffer with rank-based values
    for (int i = 0; i < nints; i++) {
        buf[i] = rank + 1;
    }

    // Create datatype
    // each process writes 1 int, skips (nprocs-1) ints, repeats
    MPI_Type_vector(nints, INTS_PER_BLK, nprocs, MPI_INT, &filetype);
    MPI_Type_commit(&filetype);

    // Open file collectively
    MPI_File_open(MPI_COMM_WORLD, "output.dat",
                  MPI_MODE_CREATE | MPI_MODE_WRONLY,
                  MPI_INFO_NULL, &fh);

    // Setup file view
    MPI_File_set_view(fh, rank * sizeof(int), MPI_INT, filetype,
                      "native", MPI_INFO_NULL);

    // Collective MPI-IO write
    MPI_File_write_all(fh, buf, nints, MPI_INT, MPI_STATUS_IGNORE);

    // Close file
    MPI_File_close(&fh);

    // Free datatype
    MPI_Type_free(&filetype);

    free(buf);

    // Finalize
    MPI_Finalize();

    return 0;
}
