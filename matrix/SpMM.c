//
// Created by kutay on 07.01.2024.
//
#include "../inc/SpMM.h"
#include <string.h>


void spmm_tp(SparseMat *A, Matrix *B, Matrix *C, TP_Comm *comm, int mode, wct *wct_time) {

    if (!comm->reducer.init) {
        switch (mode) {
            case WCT_FULL:
                spmm_tp_std(A, B, C, comm, wct_time);
                break;
            case WCT_PROFILE :
                spmm_tp_prf(A, B, C, comm, wct_time);//WCT ARRAY OF Size 3
                break;
        }
    } else {
        switch (mode) {
            case WCT_FULL:
                spmm_tp_pr(A, B, C, comm, wct_time);
                break;
            case WCT_PROFILE:
                spmm_tp_pr_prf(A, B, C, comm, wct_time);//WCT ARRAY OF Size 5
                break;
        }

    }

}

void spmm_op(SparseMat *A, Matrix *B, Matrix *C, OP_Comm *comm, int mode, wct *wct_time) {
    switch (mode) {
        case WCT_FULL: //
            spmm_op_std(A, B, C, comm, wct_time);
            break;
        case WCT_PROFILE:
            spmm_op_prf(A, B, C, comm, wct_time);//WCT ARRAY OF Size 2
            break;

    }

}

void spmm_tp_std(SparseMat *A, Matrix *B, Matrix *C, TP_Comm *comm, wct *wct_time) {
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int i, j, k;
    double t1, t2, t3;

    memset(C->entries[0], 0, C->m * C->n * sizeof(double));

    int ind, ind_c;
    int range;
    int base, part;
    MPI_Startall(comm->msgRecvCount_p1, comm->recv_ls_p1);
    MPI_Startall(comm->msgRecvCount_p2, comm->recv_ls_p2);
    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();

    for (i = 0; i < comm->msgSendCount_p1; i++) {
        part = comm->send_proc_list_p1[i];
        range = comm->sendBuffer_p1.proc_map[part + 1] - comm->sendBuffer_p1.proc_map[part];
        base = comm->sendBuffer_p1.proc_map[part];
        for (j = 0; j < range; j++) {
            ind = comm->sendBuffer_p1.row_map_lcl[base + j];
            memcpy(comm->sendBuffer_p1.buffer[base + j], B->entries[ind], sizeof(double) * B->n);
        }
        MPI_Rsend(&(comm->sendBuffer_p1.buffer[base][0]),
                  range * B->n,
                  MPI_DOUBLE,
                  part,
                  0,
                  MPI_COMM_WORLD);
        //&(Comm->send_ls_p2[i]));
    }



    //MPI_Waitall(comm->msgSendCount_p1, comm->send_ls_p1, MPI_STATUSES_IGNORE);
    MPI_Waitall(comm->msgRecvCount_p1, comm->recv_ls_p1, MPI_STATUSES_IGNORE);


    for (i = 0; i < comm->msgSendCount_p2; i++) {
        part = comm->send_proc_list_p2[i];
        range = comm->sendBuffer_p2.proc_map[part + 1] - comm->sendBuffer_p2.proc_map[part];
        base = comm->sendBuffer_p2.proc_map[part];

        for (j = 0; j < range; j++) {
            ind = comm->sendBuffer_p2.row_map_lcl[base + j];
            memcpy(comm->sendBuffer_p2.buffer[base + j], B->entries[ind], sizeof(double) * B->n);
        }
        MPI_Rsend(&(comm->sendBuffer_p2.buffer[base][0]),
                  range * B->n,
                  MPI_DOUBLE,
                  part,
                  1,
                  MPI_COMM_WORLD);
        //&(Comm->send_ls_p2[i]));
    }


    MPI_Waitall(comm->msgRecvCount_p2, comm->recv_ls_p2, MPI_STATUSES_IGNORE);
    //MPI_Waitall(comm->msgSendCount_p2, comm->send_ls_p2, MPI_STATUSES_IGNORE);


    for (i = 0; i < A->m; i++) {
        for (j = A->ia[i]; j < A->ia[i + 1]; j++) {
            int tmp = A->ja_mapped[j];
            for (k = 0; k < C->n; k++) {
                C->entries[i][k] += A->val[j] * B->entries[tmp][k];
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    t2 = MPI_Wtime();
    wct_time->total_t = t2 - t1;
}

void spmm_tp_prf(SparseMat *A, Matrix *B, Matrix *C, TP_Comm *comm, wct *wct_time) {
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int i, j, k;
    double t1, t2, t3;

    memset(C->entries[0], 0, C->m * C->n * sizeof(double));

    int ind, ind_c;
    int range;
    int base, part;

    MPI_Startall(comm->msgRecvCount_p1, comm->recv_ls_p1);
    MPI_Startall(comm->msgRecvCount_p2, comm->recv_ls_p2);
    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();


    for (i = 0; i < comm->msgSendCount_p1; i++) {
        part = comm->send_proc_list_p1[i];
        range = comm->sendBuffer_p1.proc_map[part + 1] - comm->sendBuffer_p1.proc_map[part];
        base = comm->sendBuffer_p1.proc_map[part];
        for (j = 0; j < range; j++) {
            ind = comm->sendBuffer_p1.row_map_lcl[base + j];
            memcpy(comm->sendBuffer_p1.buffer[base + j], B->entries[ind], sizeof(double) * B->n);
        }
        MPI_Rsend(&(comm->sendBuffer_p1.buffer[base][0]),
                  range * B->n,
                  MPI_DOUBLE,
                  part,
                  0,
                  MPI_COMM_WORLD);
        //&(Comm->send_ls_p2[i]));
    }



    //MPI_Waitall(comm->msgSendCount_p1, comm->send_ls_p1, MPI_STATUSES_IGNORE);
    MPI_Waitall(comm->msgRecvCount_p1, comm->recv_ls_p1, MPI_STATUSES_IGNORE);
    MPI_Barrier(MPI_COMM_WORLD);
    t2 = MPI_Wtime();
    wct_time->p1_comm_t = t2 - t1;

    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();

    for (i = 0; i < comm->msgSendCount_p2; i++) {
        part = comm->send_proc_list_p2[i];
        range = comm->sendBuffer_p2.proc_map[part + 1] - comm->sendBuffer_p2.proc_map[part];
        base = comm->sendBuffer_p2.proc_map[part];

        for (j = 0; j < range; j++) {
            ind = comm->sendBuffer_p2.row_map_lcl[base + j];
            memcpy(comm->sendBuffer_p2.buffer[base + j], B->entries[ind], sizeof(double) * B->n);
        }
        MPI_Rsend(&(comm->sendBuffer_p2.buffer[base][0]),
                  range * B->n,
                  MPI_DOUBLE,
                  part,
                  1,
                  MPI_COMM_WORLD);
        //&(Comm->send_ls_p2[i]));
    }


    MPI_Waitall(comm->msgRecvCount_p2, comm->recv_ls_p2, MPI_STATUSES_IGNORE);
    //MPI_Waitall(comm->msgSendCount_p2, comm->send_ls_p2, MPI_STATUSES_IGNORE);
    MPI_Barrier(MPI_COMM_WORLD);
    t2 = MPI_Wtime();
    wct_time->p2_comm_t = t2 - t1;
    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();
    for (i = 0; i < A->m; i++) {
        for (j = A->ia[i]; j < A->ia[i + 1]; j++) {
            int tmp = A->ja_mapped[j];
            for (k = 0; k < C->n; k++) {
                C->entries[i][k] += A->val[j] * B->entries[tmp][k];
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    t2 = MPI_Wtime();
    wct_time->SpMM_t = t2 - t1;
}

void spmm_tp_pr(SparseMat *A, Matrix *B, Matrix *C, TP_Comm *comm, wct *wct_time) {
    int i, j, k;
    int ind, ind_c;
    int range;
    int base, part;
    double t1, t2, t3;
    memset(C->entries[0], 0, C->m * C->n * sizeof(double));

    MPI_Startall(comm->msgRecvCount_p1, comm->recv_ls_p1);
    MPI_Startall(comm->msgRecvCount_p2, comm->recv_ls_p2);
    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();

    int idx, vtx, tmp;
    double factor;
    for (i = 0; i < comm->reducer.lcl_count; i++) {
        idx = comm->reducer.reduce_local[i];
        vtx = comm->reducer.reduce_list_mapped[idx];
        //This loop can be handles outside of spmm
        for (k = 0; k < C->n; k++) {
            B->entries[vtx][k] = 0;
        }
        for (int j = 1; j <= comm->reducer.reduce_source_mapped[idx][0]; j++) {
            tmp = comm->reducer.reduce_source_mapped[idx][j];
            factor = comm->reducer.reduce_factors[idx][j - 1];
            for (k = 0; k < C->n; k++) {
                B->entries[vtx][k] = B->entries[vtx][k] + B->entries[tmp][k] * factor;
            }
        }
    }
    for (i = 0; i < comm->msgSendCount_p1; i++) {
        part = comm->send_proc_list_p1[i];
        range = comm->sendBuffer_p1.proc_map[part + 1] - comm->sendBuffer_p1.proc_map[part];
        base = comm->sendBuffer_p1.proc_map[part];
        for (j = 0; j < range; j++) {
            ind = comm->sendBuffer_p1.row_map_lcl[base + j];
            memcpy(comm->sendBuffer_p1.buffer[base + j], B->entries[ind], sizeof(double) * B->n);
        }
        MPI_Rsend(&(comm->sendBuffer_p1.buffer[base][0]),
                  range * B->n,
                  MPI_DOUBLE,
                  part,
                  0,
                  MPI_COMM_WORLD);
        //&(Comm->send_ls_p2[i]));
    }
    //MPI_Waitall(comm->msgSendCount_p1, comm->send_ls_p1, MPI_STATUSES_IGNORE);
    MPI_Waitall(comm->msgRecvCount_p1, comm->recv_ls_p1, MPI_STATUSES_IGNORE);
    for (i = 0; i < comm->reducer.nlcl_count; i++) {
        idx = comm->reducer.reduce_nonlocal[i];
        vtx = comm->reducer.reduce_list_mapped[idx];
        for (k = 0; k < C->n; k++) {
            B->entries[vtx][k] = 0;
        }
        for (int j = 1; j <= comm->reducer.reduce_source_mapped[idx][0]; j++) {
            tmp = comm->reducer.reduce_source_mapped[idx][j];
            factor = comm->reducer.reduce_factors[idx][j - 1];
            for (k = 0; k < C->n; k++) {
                B->entries[vtx][k] = B->entries[vtx][k] + B->entries[tmp][k] * factor;
            }
        }
    }
    for (i = 0; i < comm->msgSendCount_p2; i++) {
        part = comm->send_proc_list_p2[i];
        range = comm->sendBuffer_p2.proc_map[part + 1] - comm->sendBuffer_p2.proc_map[part];
        base = comm->sendBuffer_p2.proc_map[part];

        for (j = 0; j < range; j++) {
            ind = comm->sendBuffer_p2.row_map_lcl[base + j];
            memcpy(comm->sendBuffer_p2.buffer[base + j], B->entries[ind], sizeof(double) * B->n);
        }
        MPI_Rsend(&(comm->sendBuffer_p2.buffer[base][0]),
                  range * B->n,
                  MPI_DOUBLE,
                  part,
                  1,
                  MPI_COMM_WORLD);
        //&(Comm->send_ls_p2[i]));
    }

    MPI_Waitall(comm->msgRecvCount_p2, comm->recv_ls_p2, MPI_STATUSES_IGNORE);
    //MPI_Waitall(comm->msgSendCount_p2, comm->send_ls_p2, MPI_STATUSES_IGNORE);
    for (i = 0; i < A->m; i++) {
        for (j = A->ia[i]; j < A->ia[i + 1]; j++) {
            int tmp = A->ja_mapped[j];
            for (k = 0; k < C->n; k++) {
                C->entries[i][k] += A->val[j] * B->entries[tmp][k];
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    t2 = MPI_Wtime();
    wct_time->total_t = t2 - t1;
}

void spmm_tp_pr_prf(SparseMat *A, Matrix *B, Matrix *C, TP_Comm *comm, wct *wct_time) {
    int i, j, k;

    int ind, ind_c;
    int range;
    int base, part;
    double t1, t2, t3;
    memset(C->entries[0], 0, C->m * C->n * sizeof(double));
    int idx, vtx, tmp;
    double factor;
    MPI_Startall(comm->msgRecvCount_p1, comm->recv_ls_p1);
    MPI_Startall(comm->msgRecvCount_p2, comm->recv_ls_p2);
    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();
    for (i = 0; i < comm->reducer.lcl_count; i++) {
        idx = comm->reducer.reduce_local[i];
        vtx = comm->reducer.reduce_list_mapped[idx];
        //This loop can be handles outside of spmm
        for (k = 0; k < C->n; k++) {
            B->entries[vtx][k] = 0;
        }
        for (int j = 1; j <= comm->reducer.reduce_source_mapped[idx][0]; j++) {
            tmp = comm->reducer.reduce_source_mapped[idx][j];
            factor = comm->reducer.reduce_factors[idx][j - 1];
            for (k = 0; k < C->n; k++) {
                B->entries[vtx][k] = B->entries[vtx][k] + B->entries[tmp][k] * factor;
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    t2 = MPI_Wtime();
    wct_time->p1_reduce_t = t2 - t1;
    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();
    for (i = 0; i < comm->msgSendCount_p1; i++) {
        part = comm->send_proc_list_p1[i];
        range = comm->sendBuffer_p1.proc_map[part + 1] - comm->sendBuffer_p1.proc_map[part];
        base = comm->sendBuffer_p1.proc_map[part];
        for (j = 0; j < range; j++) {
            ind = comm->sendBuffer_p1.row_map_lcl[base + j];
            memcpy(comm->sendBuffer_p1.buffer[base + j], B->entries[ind], sizeof(double) * B->n);
        }
        MPI_Rsend(&(comm->sendBuffer_p1.buffer[base][0]),
                  range * B->n,
                  MPI_DOUBLE,
                  part,
                  0,
                  MPI_COMM_WORLD);
        //&(Comm->send_ls_p2[i]));
    }
    //MPI_Waitall(comm->msgSendCount_p1, comm->send_ls_p1, MPI_STATUSES_IGNORE);
    MPI_Waitall(comm->msgRecvCount_p1, comm->recv_ls_p1, MPI_STATUSES_IGNORE);

    MPI_Barrier(MPI_COMM_WORLD);
    t2 = MPI_Wtime();
    wct_time->p1_comm_t = t2 - t1;

    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();
    for (i = 0; i < comm->reducer.nlcl_count; i++) {
        idx = comm->reducer.reduce_nonlocal[i];
        vtx = comm->reducer.reduce_list_mapped[idx];
        for (k = 0; k < C->n; k++) {
            B->entries[vtx][k] = 0;
        }
        for (int j = 1; j <= comm->reducer.reduce_source_mapped[idx][0]; j++) {
            tmp = comm->reducer.reduce_source_mapped[idx][j];
            factor = comm->reducer.reduce_factors[idx][j - 1];
            for (k = 0; k < C->n; k++) {
                B->entries[vtx][k] = B->entries[vtx][k] + B->entries[tmp][k] * factor;
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    t2 = MPI_Wtime();
    wct_time->post_p1_reduce_t = t2 - t1;

    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();
    for (i = 0; i < comm->msgSendCount_p2; i++) {
        part = comm->send_proc_list_p2[i];
        range = comm->sendBuffer_p2.proc_map[part + 1] - comm->sendBuffer_p2.proc_map[part];
        base = comm->sendBuffer_p2.proc_map[part];

        for (j = 0; j < range; j++) {
            ind = comm->sendBuffer_p2.row_map_lcl[base + j];
            memcpy(comm->sendBuffer_p2.buffer[base + j], B->entries[ind], sizeof(double) * B->n);
        }
        MPI_Rsend(&(comm->sendBuffer_p2.buffer[base][0]),
                  range * B->n,
                  MPI_DOUBLE,
                  part,
                  1,
                  MPI_COMM_WORLD);
        //&(Comm->send_ls_p2[i]));
    }
    MPI_Waitall(comm->msgRecvCount_p2, comm->recv_ls_p2, MPI_STATUSES_IGNORE);
    //MPI_Waitall(comm->msgSendCount_p2, comm->send_ls_p2, MPI_STATUSES_IGNORE);

    MPI_Barrier(MPI_COMM_WORLD);
    t2 = MPI_Wtime();
    wct_time->p2_comm_t = t2 - t1;

    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();
    for (i = 0; i < A->m; i++) {
        for (j = A->ia[i]; j < A->ia[i + 1]; j++) {
            int tmp = A->ja_mapped[j];
            for (k = 0; k < C->n; k++) {
                C->entries[i][k] += A->val[j] * B->entries[tmp][k];
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    t2 = MPI_Wtime();
    wct_time->SpMM_t = t2 - t1;
}


void spmm_op_std(SparseMat *A, Matrix *B, Matrix *C, OP_Comm *comm, wct *wct_time) {
    int i, j, k;
    double t1, t2, t3;

    int ind, ind_c;
    int range;
    int base, part;
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    memset(C->entries[0], 0, C->m * C->n * sizeof(double));

    MPI_Startall(comm->msgRecvCount, comm->recv_ls);
    if (world_rank == 0) {
        printf("Startall\n");
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();


    for (i = 0; i < comm->msgSendCount; i++) {
        part = comm->send_proc_list[i];
        range = comm->sendBuffer.proc_map[part + 1] - comm->sendBuffer.proc_map[part];
        base = comm->sendBuffer.proc_map[part];

        for (j = 0; j < range; j++) {
            ind = comm->sendBuffer.row_map_lcl[base + j];
            memcpy(comm->sendBuffer.buffer[base + j], B->entries[ind], sizeof(double) * B->n);
        }
        MPI_Rsend(&(comm->sendBuffer.buffer[base][0]),
                  range * B->n,
                  MPI_DOUBLE,
                  part,
                  0,
                  MPI_COMM_WORLD);
        //&(comm->send_ls[i]));
    }


    //MPI_Waitall(comm->msgSendCount, comm->send_ls, MPI_STATUSES_IGNORE);
    if (world_rank == 0) {
        printf("waiting recv\n");
        fflush(stdout);
    }
    MPI_Waitall(comm->msgRecvCount, comm->recv_ls, MPI_STATUSES_IGNORE);

    for (i = 0; i < A->m; i++) {
        for (j = A->ia[i]; j < A->ia[i + 1]; j++) {
            int tmp = A->ja_mapped[j];
            for (k = 0; k < C->n; k++) {
                C->entries[i][k] += A->val[j] * B->entries[tmp][k];
            }
        }
    }
    if (world_rank == 0) {
        printf("done\n");
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    t2 = MPI_Wtime();
    wct_time->total_t = t2 - t1;
}

void spmm_op_prf(SparseMat *A, Matrix *B, Matrix *C, OP_Comm *comm, wct *wct_time) {
    int i, j, k;
    double t1, t2, t3;

    int ind, ind_c;
    int range;
    int base, part;
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Startall(comm->msgRecvCount, comm->recv_ls);
    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();


    for (i = 0; i < comm->msgSendCount; i++) {
        part = comm->send_proc_list[i];
        range = comm->sendBuffer.proc_map[part + 1] - comm->sendBuffer.proc_map[part];
        base = comm->sendBuffer.proc_map[part];

        for (j = 0; j < range; j++) {
            ind = comm->sendBuffer.row_map_lcl[base + j];
            memcpy(comm->sendBuffer.buffer[base + j], B->entries[ind], sizeof(double) * B->n);
        }
        MPI_Rsend(&(comm->sendBuffer.buffer[base][0]),
                  range * B->n,
                  MPI_DOUBLE,
                  part,
                  0,
                  MPI_COMM_WORLD);
        //&(comm->send_ls[i]));
    }


    //MPI_Waitall(comm->msgSendCount, comm->send_ls, MPI_STATUSES_IGNORE);
    if (world_rank == 0) {
        printf("Waiting for recv\n");
        fflush(stdout);
    }
    MPI_Waitall(comm->msgRecvCount, comm->recv_ls, MPI_STATUSES_IGNORE);

    MPI_Barrier(MPI_COMM_WORLD);
    t2 = MPI_Wtime();
    wct_time->p2_comm_t = t2 - t1;

    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();

    for (i = 0; i < A->m; i++) {
        for (j = A->ia[i]; j < A->ia[i + 1]; j++) {
            int tmp = A->ja_mapped[j];
            for (k = 0; k < C->n; k++) {
                C->entries[i][k] += A->val[j] * B->entries[tmp][k];
            }
        }
    }
    if (world_rank == 0) {
        printf("Done\n");
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    t2 = MPI_Wtime();
    wct_time->SpMM_t = t2 - t1;
}


void map_csr(SparseMat *A, TP_Comm *comm) {
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int *global_map = (int *) malloc(A->gn * sizeof(int));

    for (int i = 0; i < A->gn; i++) {
        global_map[i] = -1;
    }

    for (int i = 0; i < A->m; i++) {
        global_map[A->l2gMap[i]] = i;
    }
    int base = A->m;
    for (int i = 0; i < comm->recvBuffer_p1.count; i++) {
        global_map[comm->recvBuffer_p1.row_map[i]] = base + i;
    }


    base += comm->recvBuffer_p1.count;
    for (int i = 0; i < comm->recvBuffer_p2.count; i++) {
        global_map[comm->recvBuffer_p2.row_map[i]] = base + i;
    }

    for (int i = 0; i < A->nnz; i++) {
        A->ja_mapped[i] = global_map[A->ja[i]];
        if (A->ja_mapped[i] == -1) {
            printf("Incoming seg fault 1 -> %d\n", A->ja[i]);
        }
    }

    comm->sendBuffer_p1.row_map_lcl = (int *) malloc(comm->sendBuffer_p1.count * sizeof(int));
    for (int i = 0; i < comm->sendBuffer_p1.count; ++i) {
        comm->sendBuffer_p1.row_map_lcl[i] = global_map[comm->sendBuffer_p1.row_map[i]];
        if (comm->sendBuffer_p1.row_map_lcl[i] == -1) {
            printf("Incoming seg fault 2\n");
        }
    }

    comm->sendBuffer_p2.row_map_lcl = (int *) malloc(comm->sendBuffer_p2.count * sizeof(int));
    for (int i = 0; i < comm->sendBuffer_p2.count; ++i) {
        comm->sendBuffer_p2.row_map_lcl[i] = global_map[comm->sendBuffer_p2.row_map[i]];
        if (comm->sendBuffer_p2.row_map_lcl[i] == -1) {
            printf("Incoming seg fault 3\n");
        }
    }

    comm->recvBuffer_p1.row_map_lcl = (int *) malloc(comm->recvBuffer_p1.count * sizeof(int));
    for (int i = 0; i < comm->recvBuffer_p1.count; ++i) {
        comm->recvBuffer_p1.row_map_lcl[i] = global_map[comm->recvBuffer_p1.row_map[i]];
        if (comm->recvBuffer_p1.row_map_lcl[i] == -1) {
            printf("Incoming seg fault 4\n");
        }
    }
    comm->recvBuffer_p2.row_map_lcl = (int *) malloc(comm->recvBuffer_p2.count * sizeof(int));
    for (int i = 0; i < comm->recvBuffer_p2.count; ++i) {
        comm->recvBuffer_p2.row_map_lcl[i] = global_map[comm->recvBuffer_p2.row_map[i]];
        if (comm->recvBuffer_p2.row_map_lcl[i] == -1) {
            printf("Incoming seg fault 5\n");
        }
    }

    if (comm->reducer.init) {
        int *tmp = (int *) malloc(comm->reducer.reduce_count * sizeof(int));
        int ctr = 0;
        int flag;
        for (int i = 0; i < comm->reducer.reduce_count; i++) {
            comm->reducer.reduce_list_mapped[i] = global_map[comm->reducer.reduce_list[i]];
            flag = 0;
            for (int j = 1; j <= comm->reducer.reduce_source_mapped[i][0]; j++) {
                comm->reducer.reduce_source_mapped[i][j] = global_map[comm->reducer.reduce_source_mapped[i][j]];
                if (comm->reducer.reduce_source_mapped[i][j] == -1) {
                    printf("Incoming seg fault 6\n");
                }
                if (A->inPart[comm->reducer.reduce_source_mapped[i][j]] != world_rank) {
                    flag = 1;
                }
            }
            if (flag == 1) {
                ctr++;
                tmp[i] = 1;
            } else {
                tmp[i] = 0;
            }
        }

        comm->reducer.reduce_local = (int *) malloc((comm->reducer.reduce_count - ctr) * sizeof(int));
        comm->reducer.reduce_nonlocal = (int *) malloc(ctr * sizeof(int));
        comm->reducer.lcl_count = comm->reducer.reduce_count - ctr;
        comm->reducer.nlcl_count = ctr;
        int ctr_0 = 0, ctr_1 = 0;
        for (int i = 0; i < comm->reducer.reduce_count; i++) {
            if (tmp[i] == 1) {
                comm->reducer.reduce_nonlocal[ctr_0++] = i;
            } else {
                comm->reducer.reduce_local[ctr_1++] = i;
            }
        }
    }

    free(global_map);

}

void map_csr_op(SparseMat *A, OP_Comm *comm) {
    int *global_map = (int *) malloc(A->gn * sizeof(int));

    for (int i = 0; i < A->gn; i++) {
        global_map[i] = -1;
    }

    for (int i = 0; i < A->m; i++) {
        global_map[A->l2gMap[i]] = i;
    }
    int base = A->m;
    for (int i = 0; i < comm->recvBuffer.count; i++) {
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

inline wct wct_init() {
    wct wct_time = {0, 0, 0, 0, 0, 0};
    return wct_time;
}

void wct_print(wct *wct_time) {
    if (wct_time->total_t == 0) {
        wct_time->total_t = wct_time->p1_reduce_t + wct_time->p1_comm_t + wct_time->post_p1_reduce_t +
                            wct_time->p2_comm_t + wct_time->SpMM_t;
    }
    printf(",%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n", wct_time->p1_reduce_t, wct_time->p1_comm_t, wct_time->post_p1_reduce_t,
           wct_time->p2_comm_t, wct_time->SpMM_t, wct_time->total_t);
}
