#include <stdio.h>
#include <mpi.h>
#include "inc/SparseMat.h"
#include "inc/CommHandler.h"
int main() {
    MPI_Init(NULL, NULL);
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    char* f_inpart = "/home/kutay/CLionProjects/Spmm_TaRe/ri2010/part_files/ri2010.inpart.4";
    char* f_mat = "/home/kutay/CLionProjects/Spmm_TaRe/ri2010/part_files/ri2010.inpart.4.bin";
    char* f_comm = "/home/kutay/CLionProjects/Spmm_TaRe/ri2010/part_files/ri2010-4-phases.bin";
    printf("Hello, World! %d %d\n", world_size, world_rank);

    SparseMat *A = readSparseMat(f_mat, STORE_BY_ROWS, f_inpart);
    //printf("A->m: %d\n", A->m);
    //printf("A->n: %d\n", A->n);
    TP_Comm *comm = readTwoPhaseComm(f_comm, 1);





    sparseMatFree(A);
    MPI_Finalize();;
    return 0;
}
