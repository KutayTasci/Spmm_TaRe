//
// Created by kutay on 07.01.2024.
//
#include "../inc/SpMM.h"
#include <string.h>


void spmm_tp(SparseMat* A, Matrix* B, Matrix* C, TP_Comm* comm) {
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int i, j, k;

    printf("world rank: %d\n", world_rank);
    MPI_Barrier(MPI_COMM_WORLD);

    memset(C->entries[0], 0, C->m * C->n * sizeof(double));
    MPI_Request* request_send_p1 = (MPI_Request*) malloc((comm->msgSendCount_p1) * sizeof(MPI_Request));
    MPI_Request* request_recv_p1 = (MPI_Request*) malloc((comm->msgRecvCount_p1) * sizeof(MPI_Request));
    MPI_Status* status_list_r_p1 = (MPI_Status*) malloc((comm->msgRecvCount_p1) * sizeof(MPI_Status));
    MPI_Status* status_list_s_p1 = (MPI_Status*) malloc((comm->msgSendCount_p1) * sizeof(MPI_Status));

    MPI_Request* request_send_p2 = (MPI_Request*) malloc((comm->msgSendCount_p2) * sizeof(MPI_Request));
    MPI_Request* request_recv_p2 = (MPI_Request*) malloc((comm->msgRecvCount_p2) * sizeof(MPI_Request));
    MPI_Status* status_list_r_p2 = (MPI_Status*) malloc((comm->msgRecvCount_p2) * sizeof(MPI_Status));
    MPI_Status* status_list_s_p2 = (MPI_Status*) malloc((comm->msgSendCount_p2) * sizeof(MPI_Status));

    int ind, ind_c;
    int range;
    int base;

    ind_c=0;
    for (i = 0;i < world_size; i++) {
        if (i != world_rank) {
            range = comm->recvBuffer_p1.proc_map[i+1] - comm->recvBuffer_p1.proc_map[i];
            base = B->phase_1 + comm->recvBuffer_p1.proc_map[i];
            if (range != 0) {

                MPI_Irecv(&(B->entries[base][0]),
                          range * B->n,
                          MPI_DOUBLE,
                          i,
                          0,
                          MPI_COMM_WORLD,
                          &(request_recv_p1[ind_c]));
                ind_c++;

            }
        }
    }

    ind_c = 0;
    for (i=0;i < world_size; i++) {
        range = comm->sendBuffer_p1.proc_map[i+1] - comm->sendBuffer_p1.proc_map[i];
        base = comm->sendBuffer_p1.proc_map[i];
        if (i != world_rank) {
            if (range != 0) {
                for (j = 0;j < range; j++) {
                    ind = comm->sendBuffer_p1.row_map_lcl[base + j];
                    memcpy(comm->sendBuffer_p1.buffer[base + j],  B->entries[ind] , sizeof(double) * B->n);
                }
                MPI_Isend(&(comm->sendBuffer_p1.buffer[base][0]),
                          range * B->n,
                          MPI_DOUBLE,
                          i,
                          0,
                          MPI_COMM_WORLD,
                          &(request_send_p1[ind_c]));
                ind_c++;
            }
        }
    }

    MPI_Waitall(comm->msgRecvCount_p1, request_recv_p1, status_list_r_p1);
    MPI_Waitall(comm->msgSendCount_p1, request_send_p1, status_list_s_p1);


    ind_c=0;
    for (i = 0;i < world_size; i++) {
        if (i != world_rank) {
            range = comm->recvBuffer_p2.proc_map[i+1] - comm->recvBuffer_p2.proc_map[i];
            base = B->phase_2 + comm->recvBuffer_p2.proc_map[i];
            if (range != 0) {

                MPI_Irecv(&(B->entries[base][0]),
                          range * B->n,
                          MPI_DOUBLE,
                          i,
                          1,
                          MPI_COMM_WORLD,
                          &(request_recv_p2[ind_c]));
                ind_c++;

            }
        }
    }

    ind_c = 0;
    for (i=0;i < world_size; i++) {
        range = comm->sendBuffer_p2.proc_map[i+1] - comm->sendBuffer_p2.proc_map[i];
        base = comm->sendBuffer_p2.proc_map[i];
        if (i != world_rank) {

            if (range != 0) {

                for (j = 0;j < range; j++) {
                    ind = comm->sendBuffer_p2.row_map_lcl[base + j];
                    memcpy(comm->sendBuffer_p2.buffer[base + j],  B->entries[ind] , sizeof(double) * B->n);

                }
                MPI_Isend(&(comm->sendBuffer_p2.buffer[base][0]),
                          range * B->n,
                          MPI_DOUBLE,
                          i,
                          1,
                          MPI_COMM_WORLD,
                          &(request_send_p2[ind_c]));
                ind_c++;

            }
        }
    }


    MPI_Waitall(comm->msgRecvCount_p2, request_recv_p2, status_list_r_p2);
    MPI_Waitall(comm->msgSendCount_p2, request_send_p2, status_list_s_p2);

    for(i=0;i<A->m;i++) {
        for (j=A->ia[i];j<A->ia[i+1];j++) {
            int tmp = A->ja_mapped[j];
            for (k = 0; k<C->n;k++) {
                C->entries[i][k] += A->val[j] * B->entries[tmp][k];
            }
        }
    }

}

void map_csr(SparseMat *A, TP_Comm *comm) {
    int* global_map = (int *) malloc(A->gn * sizeof(int));

    for (int i = 0; i < A->m; i++) {
        global_map[A->l2gMap[i]] = i;
    }
    int base = A->m;
    for (int i = 0;i < comm->recvBuffer_p1.count; i++) {
        global_map[comm->recvBuffer_p1.row_map[i]] = base + i;
    }

    base += comm->recvBuffer_p1.count;
    for (int i = 0;i < comm->recvBuffer_p2.count; i++) {
        global_map[comm->recvBuffer_p2.row_map[i]] = base + i;
    }

    for (int i = 0; i < A->nnz; i++) {
        A->ja_mapped[i] = global_map[A->ja[i]];
    }

    comm->sendBuffer_p1.row_map_lcl = (int *) malloc(comm->sendBuffer_p1.count * sizeof(int));
    for (int i = 0; i < comm->sendBuffer_p1.count; ++i) {
        comm->sendBuffer_p1.row_map_lcl[i] = global_map[comm->sendBuffer_p1.row_map[i]];
    }

    comm->sendBuffer_p2.row_map_lcl = (int *) malloc(comm->sendBuffer_p2.count * sizeof(int));
    for (int i = 0; i < comm->sendBuffer_p2.count; ++i) {
        comm->sendBuffer_p2.row_map_lcl[i] = global_map[comm->sendBuffer_p2.row_map[i]];
    }

    comm->recvBuffer_p1.row_map_lcl = (int *) malloc(comm->recvBuffer_p1.count * sizeof(int));
    for (int i = 0; i < comm->recvBuffer_p1.count; ++i) {
        comm->recvBuffer_p1.row_map_lcl[i] = global_map[comm->recvBuffer_p1.row_map[i]];
    }
    comm->recvBuffer_p2.row_map_lcl = (int *) malloc(comm->recvBuffer_p2.count * sizeof(int));
    for (int i = 0; i < comm->recvBuffer_p2.count; ++i) {
        comm->recvBuffer_p2.row_map_lcl[i] = global_map[comm->recvBuffer_p2.row_map[i]];
    }

    free(global_map);

}