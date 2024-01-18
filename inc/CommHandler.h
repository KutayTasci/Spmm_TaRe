//
// Created by kutay on 20.11.2023.
//

#ifndef SPMM_TARE_COMMHANDLER_H
#define SPMM_TARE_COMMHANDLER_H

typedef struct {
    int *proc_map; //world_size+1
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
    CommBuffer recvBuffer;
    int msgRecvCount;
} OP_Comm;

/*
 * Two phase communication data structure added by @Kutay
 */
typedef struct {
    CommBuffer sendBuffer_p1;
    int msgSendCount_p1;
    CommBuffer recvBuffer_p1;
    int msgRecvCount_p1;
    CommBuffer sendBuffer_p2;
    int msgSendCount_p2;
    CommBuffer recvBuffer_p2;
    int msgRecvCount_p2;
} TP_Comm;

void CommBufferInit(CommBuffer*  buff);
void CommBufferFree(CommBuffer* buff);
TP_Comm* readTwoPhaseComm(char* fName, int f);
OP_Comm* readOnePhaseComm(char* fName, int f);

#endif //SPMM_TARE_COMMHANDLER_H
