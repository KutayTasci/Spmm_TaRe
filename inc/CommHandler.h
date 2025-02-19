//
// Created by kutay on 20.11.2023.
//

#ifndef SPMM_TARE_COMMHANDLER_H
#define SPMM_TARE_COMMHANDLER_H

#include <mpi.h>
#include <stdbool.h>

#ifdef u64
typedef unsigned long long udx_t;
#else
typedef unsigned int udx_t;
#endif

typedef struct {
    udx_t *proc_map; //world_size+1
    int *row_map; //count
    int *row_map_lcl; //count
    double **buffer; //count * f
    int count;
    int f;
} CommBuffer;

/*
 * One phase communication data structure added by @Kutay
 */
typedef struct {
    CommBuffer sendBuffer;
    int msgSendCount;
    MPI_Request *send_ls;
    int *send_proc_list;

    CommBuffer recvBuffer;
    int msgRecvCount;
    MPI_Request *recv_ls;
    int *recv_proc_list;
} OP_Comm;

/*
 * Two phase communication data structure added by @Kutay
 */

typedef struct {
    bool init;

    int reduce_count;
    int *reduce_list;
    int *reduce_list_mapped;
    int **reduce_source_mapped; // first element is the length of the list
    double **reduce_factors;
    int *reduce_local;
    int lcl_count;
    int *reduce_nonlocal;
    int nlcl_count;
} Reducer;

typedef struct {
    Reducer reducer;

    CommBuffer sendBuffer_p1;
    int msgSendCount_p1;
    MPI_Request *send_ls_p1;
    int *send_proc_list_p1;

    CommBuffer recvBuffer_p1;
    int msgRecvCount_p1;
    MPI_Request *recv_ls_p1;
    int *recv_proc_list_p1;

    CommBuffer sendBuffer_p2;
    int msgSendCount_p2;
    MPI_Request *send_ls_p2;
    int *send_proc_list_p2;

    CommBuffer recvBuffer_p2;
    int msgRecvCount_p2;
    MPI_Request *recv_ls_p2;
    int *recv_proc_list_p2;
} TP_Comm;

void CommBufferInit(CommBuffer *buff);

void CommBufferFree(CommBuffer *buff);

TP_Comm *readTwoPhaseComm(char *fName, int f, bool partial_reduce);

OP_Comm *readOnePhaseComm(char *fName, int f);

void prep_comm_tp(TP_Comm *Comm);

void prep_comm_op(OP_Comm *Comm);

#endif //SPMM_TARE_COMMHANDLER_H
