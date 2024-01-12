#include <stdio.h>
#include <mpi.h>
#include "inc/SparseMat.h"
#include "inc/CommHandler.h"
#include "inc/DenseMat.h"
#include "inc/SpMM.h"

int main() {
    MPI_Init(NULL, NULL);
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    char* f_inpart = "/home/kutay/CLionProjects/Spmm_TaRe/ri2010/part_files/KarateClub.inpart.4";
    char* f_mat = "/home/kutay/CLionProjects/Spmm_TaRe/ri2010/part_files/KarateClub.inpart.4.bin";
    char* f_comm = "/home/kutay/CLionProjects/Spmm_TaRe/ri2010/part_files/KarateClub.phases.4.bin";

    int k = 4;

    SparseMat *A = readSparseMat(f_mat, STORE_BY_ROWS, f_inpart);
    TP_Comm *comm = readTwoPhaseComm(f_comm, k);
    Matrix *X = matrix_create_tp(A->m, k, A->gn, k, comm);
    matrix_fill_double(X, 1.0);
    Matrix *Y = matrix_create_tp(A->m, k, A->gn, k, comm);

    map_csr(A, comm);

    //print communication sizes for both phases
    printf("Phase 1 send count: %d\n", comm->sendBuffer_p1.count);
    printf("Phase 1 recv count: %d\n", comm->recvBuffer_p1.count);
    printf("Phase 2 send count: %d\n", comm->sendBuffer_p2.count);
    printf("Phase 2 recv count: %d\n", comm->recvBuffer_p2.count);


    spmm_tp(A, X, Y, comm);

    /*
    for (int i = 0; i < Y->lcl_m; i++) {
        for (int j = 0; j < Y->n; j++) {
            printf("%f ", Y->entries[i][j]);
        }
        printf("\n");
    }*/

    MPI_Barrier(MPI_COMM_WORLD);
    //matrix_free(X);
    //sparseMatFree(A);
    MPI_Finalize();;
    return 0;
}
