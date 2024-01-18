//
// Created by kutay on 20.11.2023.
//
#include "../inc/CommHandler.h"
#include <stdlib.h>
#include <mpi.h>
#include <stdio.h>

/*
 * Reading CSR Matrix in parallel
 * Handles memory allocation
 * File format follows the output format of "Util" code by Oguz Selvitopi
 * Added by @Kutay
*/

/*
 * Communication buffer data structure added by @Kutay
 */
void CommBufferInit(CommBuffer*  buff) {
    double* data = (double *) malloc(buff->count * buff->f * sizeof(double));
    buff->buffer = (double **) malloc(buff->count * sizeof(double*));

    for (int i = 0; i < buff->count; i++) {
        buff->buffer[i] = &(data[buff->f*i]);
    }

}

/*
 * Communication buffer data structure added by @Kutay
 */
void CommBufferFree(CommBuffer* buff) {
    free(buff->proc_map);
    free(buff->row_map);
    free(buff->buffer[0]);
    free(buff->buffer);
    free(buff);
    buff = NULL;
}

/*
 * parallel read of two phase communication data structure
 * Added by @Kutay
 */
TP_Comm* readTwoPhaseComm(char* fName, int f) {
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int64_t sloc;
    TP_Comm *Comm = (TP_Comm *) malloc(sizeof (TP_Comm));
    FILE *fpmat = fopen(fName, "rb");

    fseek(fpmat, (world_rank*sizeof(int64_t)), SEEK_SET);
    fread(&sloc, sizeof(int64_t), 1, fpmat);

    fseek(fpmat, sloc, SEEK_SET);
    fread(&(Comm->sendBuffer_p1.count), sizeof(int), 1, fpmat);
    fread(&(Comm->recvBuffer_p1.count), sizeof(int), 1, fpmat);
    fread(&(Comm->sendBuffer_p2.count), sizeof(int), 1, fpmat);
    fread(&(Comm->recvBuffer_p2.count), sizeof(int), 1, fpmat);

    Comm->sendBuffer_p1.proc_map = (int *) malloc( (world_size + 1)* sizeof(int));
    Comm->sendBuffer_p1.row_map = (int *) malloc( Comm->sendBuffer_p1.count* sizeof(int));

    Comm->recvBuffer_p1.proc_map = (int *) malloc( (world_size + 1)* sizeof(int));
    Comm->recvBuffer_p1.row_map = (int *) malloc( Comm->recvBuffer_p1.count* sizeof(int));

    Comm->sendBuffer_p2.proc_map = (int *) malloc( (world_size + 1)* sizeof(int));
    Comm->sendBuffer_p2.row_map = (int *) malloc( Comm->sendBuffer_p2.count* sizeof(int));

    Comm->recvBuffer_p2.proc_map = (int *) malloc( (world_size + 1)* sizeof(int));
    Comm->recvBuffer_p2.row_map = (int *) malloc( Comm->recvBuffer_p2.count* sizeof(int));

    fread(Comm->sendBuffer_p1.proc_map, sizeof(int), world_size+1, fpmat);
    fread(Comm->recvBuffer_p1.proc_map, sizeof(int), world_size+1, fpmat);
    fread(Comm->sendBuffer_p2.proc_map, sizeof(int), world_size+1, fpmat);
    fread(Comm->recvBuffer_p2.proc_map, sizeof(int), world_size+1, fpmat);

    Comm->msgRecvCount_p1 = 0;
    Comm->msgRecvCount_p2 = 0;
    Comm->msgSendCount_p1 = 0;
    Comm->msgSendCount_p2 = 0;

    for (int i = 1; i <= world_size ; ++i) {
        if (Comm->sendBuffer_p1.proc_map[i] - Comm->sendBuffer_p1.proc_map[i-1] != 0) {
            Comm->msgSendCount_p1++;
        }
        if (Comm->sendBuffer_p2.proc_map[i] - Comm->sendBuffer_p2.proc_map[i-1] != 0) {
            Comm->msgSendCount_p2++;
        }
        if (Comm->recvBuffer_p1.proc_map[i] - Comm->recvBuffer_p1.proc_map[i-1] != 0) {
            Comm->msgRecvCount_p1++;
        }
        if (Comm->recvBuffer_p2.proc_map[i] - Comm->recvBuffer_p2.proc_map[i-1] != 0) {
            Comm->msgRecvCount_p2++;
        }
    }

    fread(Comm->sendBuffer_p1.row_map, sizeof(int), Comm->sendBuffer_p1.count, fpmat);
    fread(Comm->recvBuffer_p1.row_map, sizeof(int), Comm->recvBuffer_p1.count, fpmat);
    fread(Comm->sendBuffer_p2.row_map, sizeof(int), Comm->sendBuffer_p2.count, fpmat);
    fread(Comm->recvBuffer_p2.row_map, sizeof(int), Comm->recvBuffer_p2.count, fpmat);

    Comm->sendBuffer_p1.f = f;
    Comm->recvBuffer_p1.f = f;
    Comm->sendBuffer_p2.f = f;
    Comm->recvBuffer_p2.f = f;

    CommBufferInit(&(Comm->sendBuffer_p1));
    CommBufferInit(&(Comm->sendBuffer_p2));

    return Comm;
}

/*
 * parallel read of one phase communication data structure
 * Added by @Kutay
 */
OP_Comm* readOnePhaseComm(char* fName, int f) {
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int64_t sloc;
    OP_Comm *Comm = (OP_Comm *) malloc(sizeof (OP_Comm));
    FILE *fpmat = fopen(fName, "rb");
    fseek(fpmat, (world_rank*sizeof(int64_t)), SEEK_SET);
    fread(&sloc, sizeof(int64_t), 1, fpmat);

    fseek(fpmat, sloc, SEEK_SET);
    fread(&(Comm->sendBuffer.count), sizeof(int), 1, fpmat);
    fread(&(Comm->recvBuffer.count), sizeof(int), 1, fpmat);

    Comm->sendBuffer.proc_map = (int *) malloc( (world_size + 1)* sizeof(int));
    Comm->sendBuffer.row_map = (int *) malloc( Comm->sendBuffer.count* sizeof(int));

    Comm->recvBuffer.proc_map = (int *) malloc( (world_size + 1)* sizeof(int));
    Comm->recvBuffer.row_map = (int *) malloc( Comm->recvBuffer.count* sizeof(int));

    fread(Comm->sendBuffer.proc_map, sizeof(int), world_size+1, fpmat);
    fread(Comm->recvBuffer.proc_map, sizeof(int), world_size+1, fpmat);

    fread(Comm->sendBuffer.row_map, sizeof(int), Comm->sendBuffer.count, fpmat);
    fread(Comm->recvBuffer.row_map, sizeof(int), Comm->recvBuffer.count, fpmat);

    Comm->msgRecvCount = 0;
    Comm->msgSendCount = 0;

    for (int i = 1; i <= world_size ; ++i) {
        if (Comm->sendBuffer.proc_map[i] - Comm->sendBuffer.proc_map[i-1] != 0) {
            Comm->msgSendCount++;
        }
        if (Comm->recvBuffer.proc_map[i] - Comm->recvBuffer.proc_map[i-1] != 0) {
            Comm->msgRecvCount++;
        }
    }

    Comm->sendBuffer.f = f;
    Comm->recvBuffer.f = f;

    CommBufferInit(&(Comm->sendBuffer));


    return Comm;
}