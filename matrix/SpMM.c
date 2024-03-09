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

    memset(C->entries[0], 0, C->m * C->n * sizeof(double));

    int ind, ind_c;
    int range;
    int base, part;

    for (i = 0;i < comm->msgRecvCount_p1; i++) {
        part = comm->recv_proc_list_p1[i];
        range = comm->recvBuffer_p1.proc_map[part+1] - comm->recvBuffer_p1.proc_map[part];
        base = B->phase_1 + comm->recvBuffer_p1.proc_map[part];

        MPI_Irecv(&(B->entries[base][0]),
                  range * B->n,
                  MPI_DOUBLE,
                  part,
                  0,
                  MPI_COMM_WORLD,
                  &(comm->recv_ls_p1[i]));


    }


    for (i=0;i < comm->msgSendCount_p1; i++) {
        part = comm->send_proc_list_p1[i];
        range = comm->sendBuffer_p1.proc_map[part+1] - comm->sendBuffer_p1.proc_map[part];
        base = comm->sendBuffer_p1.proc_map[part];

        for (j = 0;j < range; j++) {
            ind = comm->sendBuffer_p1.row_map_lcl[base + j];
            memcpy(comm->sendBuffer_p1.buffer[base + j],  B->entries[ind] , sizeof(double) * B->n);
        }
        MPI_Isend(&(comm->sendBuffer_p1.buffer[base][0]),
                  range * B->n,
                  MPI_DOUBLE,
                  part,
                  0,
                  MPI_COMM_WORLD,
                  &(comm->send_ls_p1[i]));

    }

    MPI_Waitall(comm->msgRecvCount_p1, comm->recv_ls_p1, MPI_STATUSES_IGNORE);
    MPI_Waitall(comm->msgSendCount_p1, comm->send_ls_p1, MPI_STATUSES_IGNORE);

    for (i = 0;i < comm->msgRecvCount_p2; i++) {
        part = comm->recv_proc_list_p2[i];
        range = comm->recvBuffer_p2.proc_map[part+1] - comm->recvBuffer_p2.proc_map[part];
        base = B->phase_2 + comm->recvBuffer_p2.proc_map[part];

        MPI_Irecv(&(B->entries[base][0]),
                  range * B->n,
                  MPI_DOUBLE,
                  part,
                  0,
                  MPI_COMM_WORLD,
                  &(comm->recv_ls_p2[i]));


    }

    for (i=0;i < comm->msgSendCount_p2; i++) {
        part = comm->send_proc_list_p2[i];
        range = comm->sendBuffer_p2.proc_map[part+1] - comm->sendBuffer_p2.proc_map[part];
        base = comm->sendBuffer_p2.proc_map[part];

        for (j = 0;j < range; j++) {
            ind = comm->sendBuffer_p2.row_map_lcl[base + j];
            memcpy(comm->sendBuffer_p2.buffer[base + j],  B->entries[ind] , sizeof(double) * B->n);
        }
        MPI_Isend(&(comm->sendBuffer_p2.buffer[base][0]),
                  range * B->n,
                  MPI_DOUBLE,
                  part,
                  0,
                  MPI_COMM_WORLD,
                  &(comm->send_ls_p2[i]));

    }

    MPI_Waitall(comm->msgRecvCount_p2, comm->recv_ls_p2, MPI_STATUSES_IGNORE);
    MPI_Waitall(comm->msgSendCount_p2, comm->send_ls_p2, MPI_STATUSES_IGNORE);

    for(i=0;i<A->m;i++) {
        for (j=A->ia[i];j<A->ia[i+1];j++) {
            int tmp = A->ja_mapped[j];
            for (k = 0; k<C->n;k++) {
                C->entries[i][k] += A->val[j] * B->entries[tmp][k];
            }
        }
    }

}

void spmm_op(SparseMat* A, Matrix* B, Matrix* C, OP_Comm* comm) {
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int i, j, k;


    memset(C->entries[0], 0, C->m * C->n * sizeof(double));
    MPI_Request* request_send = comm->send_ls;
    MPI_Request* request_recv = comm->recv_ls;

    int ind, ind_c;
    int range;
    int base, part;


    for (i = 0;i < comm->msgRecvCount; i++) {
        part = comm->recv_proc_list[i];
        range = comm->recvBuffer.proc_map[part+1] - comm->recvBuffer.proc_map[part];
        base = B->phase_1 + comm->recvBuffer.proc_map[part];

        MPI_Irecv(&(B->entries[base][0]),
                  range * B->n,
                  MPI_DOUBLE,
                  part,
                  0,
                  MPI_COMM_WORLD,
                  &(comm->recv_ls[i]));

    }

    ind_c = 0;
    for (i=0;i < comm->msgSendCount; i++) {
        part = comm->send_proc_list[i];
        range = comm->sendBuffer.proc_map[part + 1] - comm->sendBuffer.proc_map[part];
        base = comm->sendBuffer.proc_map[part];

        for (j = 0; j < range; j++) {
            ind = comm->sendBuffer.row_map_lcl[base + j];
            memcpy(comm->sendBuffer.buffer[base + j], B->entries[ind], sizeof(double) * B->n);
        }
        MPI_Isend(&(comm->sendBuffer.buffer[base][0]),
                  range * B->n,
                  MPI_DOUBLE,
                  part,
                  0,
                  MPI_COMM_WORLD,
                  &(comm->send_ls[i]));


    }

    MPI_Waitall(comm->msgRecvCount, comm->send_ls, MPI_STATUSES_IGNORE);
    MPI_Waitall(comm->msgSendCount, comm->recv_ls, MPI_STATUSES_IGNORE);

    for(i=0;i<A->m;i++) {
        for (j=A->ia[i];j<A->ia[i+1];j++) {
            int tmp = A->ja_mapped[j];
            for (k = 0; k<C->n;k++) {
                C->entries[i][k] += A->val[j] * B->entries[tmp][k];
            }
        }
    }
}

void spmm_reduce_tp(SparseMat* A, Matrix* B, Matrix* C, TP_Comm* comm) {
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int i, j, k;

    memset(C->entries[0], 0, C->m * C->n * sizeof(double));

    for(i=0;i<A->m;i++) {
        for (j=A->ia[i];j<A->ia[i+1];j++) {
            int tmp = A->ja_mapped[j];
            for (k = 0; k<C->n;k++) {
                C->entries[tmp][k] += A->val[j] * B->entries[i][k];
            }
        }
    }

    int ind, ind_c;
    int range;
    int base, part;
	

    for (i=0;i < comm->msgSendCount_p2; i++) {
    	part = comm->send_proc_list_p2[i];
        range = comm->sendBuffer_p2.proc_map[part+1] - comm->sendBuffer_p2.proc_map[part];
        base = comm->sendBuffer_p2.proc_map[part];
        MPI_Irecv(&(comm->sendBuffer_p2.buffer[base][0]),
                  range * C->n,
                  MPI_DOUBLE,
                  part,
                  1,
                  MPI_COMM_WORLD,
                  &(comm->send_ls_p2[i]));
    }


    for (i = 0;i < comm->msgRecvCount_p2; i++) {
    	part = comm->recv_proc_list_p2[i];
        range = comm->recvBuffer_p2.proc_map[part+1] - comm->recvBuffer_p2.proc_map[part];
        base = C->phase_2 + comm->recvBuffer_p2.proc_map[part];
        MPI_Isend(&(C->entries[base][0]),
                  range * C->n,
                  MPI_DOUBLE,
                  part,
                  1,
                  MPI_COMM_WORLD,
                  &(comm->recv_ls_p2[i]));
    }
	
    MPI_Waitall(comm->msgSendCount_p2, comm->send_ls_p2, MPI_STATUSES_IGNORE);
    MPI_Waitall(comm->msgRecvCount_p2, comm->recv_ls_p2, MPI_STATUSES_IGNORE);

    for (i=0;i < comm->msgSendCount_p2; i++) {
    	part = comm->send_proc_list_p2[i];
        range = comm->sendBuffer_p2.proc_map[part+1] - comm->sendBuffer_p2.proc_map[part];
        base = comm->sendBuffer_p2.proc_map[part];
        for (j = 0;j < range; j++) {
            ind = comm->sendBuffer_p2.row_map_lcl[base + j];
            for (k = 0; k<C->n;k++) {
                C->entries[ind][k] += comm->sendBuffer_p2.buffer[base + j][k];
            }
        }
    }


    for (i=0;i < comm->msgSendCount_p1; i++) {
    	part = comm->send_proc_list_p1[i];
        range = comm->sendBuffer_p1.proc_map[part+1] - comm->sendBuffer_p1.proc_map[part];
        base = comm->sendBuffer_p1.proc_map[part];
        MPI_Irecv(&(comm->sendBuffer_p1.buffer[base][0]),
                  range * C->n,
                  MPI_DOUBLE,
                  part,
                  1,
                  MPI_COMM_WORLD,
                  &(comm->send_ls_p1[i]));
    }

    for (i = 0;i < comm->msgRecvCount_p1; i++) {
    	part = comm->recv_proc_list_p1[i];
        range = comm->recvBuffer_p1.proc_map[part+1] - comm->recvBuffer_p1.proc_map[part];
        base = C->phase_1 + comm->recvBuffer_p1.proc_map[part];
        MPI_Isend(&(C->entries[base][0]),
                  range * C->n,
                  MPI_DOUBLE,
                  part,
                  1,
                  MPI_COMM_WORLD,
                  &(comm->recv_ls_p1[i]));
    }

    MPI_Waitall(comm->msgSendCount_p1, comm->send_ls_p1, MPI_STATUSES_IGNORE);
    MPI_Waitall(comm->msgRecvCount_p1, comm->recv_ls_p1, MPI_STATUSES_IGNORE);
	
    for (i=0;i < comm->msgSendCount_p1; i++) {
    	part = comm->send_proc_list_p1[i];
        range = comm->sendBuffer_p1.proc_map[part+1] - comm->sendBuffer_p1.proc_map[part];
        base = comm->sendBuffer_p1.proc_map[part];
        for (j = 0;j < range; j++) {
            ind = comm->sendBuffer_p1.row_map_lcl[base + j];
            for (k = 0; k<C->n;k++) {
                C->entries[ind][k] += comm->sendBuffer_p1.buffer[base + j][k];
            }
        }

    }

}

void spmm_reduce_op(SparseMat* A, Matrix* B, Matrix* C, OP_Comm* comm) {
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int i, j, k;

    memset(C->entries[0], 0, C->m * C->n * sizeof(double));

    for(i=0;i<A->m;i++) {
        for (j=A->ia[i];j<A->ia[i+1];j++) {
            int tmp = A->ja_mapped[j];
            for (k = 0; k<C->n;k++) {
                C->entries[tmp][k] += A->val[j] * B->entries[i][k];
            }
        }
    }

    int ind, ind_c;
    int range;
    int base, part;

    for (i=0;i < comm->msgSendCount; i++) {
    	part = comm->send_proc_list[i];
        range = comm->sendBuffer.proc_map[part+1] - comm->sendBuffer.proc_map[part];
        base = comm->sendBuffer.proc_map[part];
        MPI_Irecv(&(comm->sendBuffer.buffer[base][0]),
                  range * C->n,
                  MPI_DOUBLE,
                  part,
                  1,
                  MPI_COMM_WORLD,
                  &(comm->send_ls[i]));

    }

    for (i = 0;i < comm->msgRecvCount; i++) {
    	part = comm->recv_proc_list[i];
        range = comm->recvBuffer.proc_map[part+1] - comm->recvBuffer.proc_map[part];
        base = C->phase_1 + comm->recvBuffer.proc_map[part];
        MPI_Isend(&(C->entries[base][0]),
                  range * C->n,
                  MPI_DOUBLE,
                  part,
                  1,
                  MPI_COMM_WORLD,
                  &(comm->recv_ls[i]));

    }

    MPI_Waitall(comm->msgRecvCount, comm->recv_ls, MPI_STATUSES_IGNORE);
    MPI_Waitall(comm->msgSendCount, comm->send_ls, MPI_STATUSES_IGNORE);

    for (i=0;i < comm->msgSendCount; i++) {
    	part = comm->send_proc_list[i];
        range = comm->sendBuffer.proc_map[part+1] - comm->sendBuffer.proc_map[part];
        base = comm->sendBuffer.proc_map[part];
        for (j = 0;j < range; j++) {
            ind = comm->sendBuffer.row_map_lcl[base + j];
            for (k = 0; k<C->n;k++) {
                C->entries[ind][k] += comm->sendBuffer.buffer[base + j][k];
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

void map_csr_op(SparseMat *A, OP_Comm *comm) {
    int* global_map = (int *) malloc(A->gn * sizeof(int));

    for (int i = 0; i < A->m; i++) {
        global_map[A->l2gMap[i]] = i;
    }
    int base = A->m;
    for (int i = 0;i < comm->recvBuffer.count; i++) {
        global_map[comm->recvBuffer.row_map[i]] = base + i;
    }

    for (int i = 0; i < A->nnz; i++) {
        A->ja_mapped[i] = global_map[A->ja[i]];
    }

    comm->sendBuffer.row_map_lcl = (int *) malloc(comm->sendBuffer.count * sizeof(int));
    for (int i = 0; i < comm->sendBuffer.count; ++i) {
        comm->sendBuffer.row_map_lcl[i] = global_map[comm->sendBuffer.row_map[i]];
    }

    comm->recvBuffer.row_map_lcl = (int *) malloc(comm->recvBuffer.count * sizeof(int));
    for (int i = 0; i < comm->recvBuffer.count; ++i) {
        comm->recvBuffer.row_map_lcl[i] = global_map[comm->recvBuffer.row_map[i]];
    }

    free(global_map);

}
