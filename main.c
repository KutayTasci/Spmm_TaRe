#include <stdio.h>
#include <mpi.h>
#include "inc/SparseMat.h"
#include "inc/CommHandler.h"
int main() {
    MPI_Init(NULL, NULL);
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    char* f_inpart = "/home/serdar/Code/PycharmProjects/cs490/out/citationCiteseer.inpart.8";
    char* f_mat = "/home/serdar/Code/PycharmProjects/cs490/out/citationCiteseer.inpart.8.bin";
    char* f_comm = "/home/serdar/Code/PycharmProjects/cs490/out/citationCiteseer.phases.8.bin";
    printf("Hello, World! %d %d\n", world_size, world_rank);

    SparseMat *A = readSparseMat(f_mat, STORE_BY_ROWS, f_inpart);
    TP_Comm *comm = readTwoPhaseComm(f_comm, 1);

    sparseMatFree(A);
    MPI_Finalize();;
    return 0;
}
