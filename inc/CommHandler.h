//
// Created by kutay on 20.11.2023.
//

#ifndef SPMM_TARE_COMMHANDLER_H
#define SPMM_TARE_COMMHANDLER_H

typedef struct {
    int *proc_map; //world_size+1
    int *row_map; //count
    double **buffer; //count * f
    int count;
    int f;
} CommBuffer;

/*
 * One phase communication data structure added by @Kutay
 */
typedef struct {
    CommBuffer sendBuffer;
    CommBuffer recvBuffer;
} OP_Comm;

/*
 * Two phase communication data structure added by @Kutay
 */
typedef struct {
    CommBuffer sendBuffer_p1;
    CommBuffer recvBuffer_p1;
    CommBuffer sendBuffer_p2;
    CommBuffer recvBuffer_p2;
} TP_Comm;

CommBuffer* CommBufferInit(int count, int f);
void CommBufferFree(CommBuffer* buff);
TP_Comm* readTwoPhaseComm(char* fName, int f);
OP_Comm* readOnePhaseComm(char* fName);

#endif //SPMM_TARE_COMMHANDLER_H
