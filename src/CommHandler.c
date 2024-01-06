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
CommBuffer *CommBufferInit(int count, int f) {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    CommBuffer *buff = (CommBuffer *) malloc(sizeof(CommBuffer));
    buff->count = count;
    buff->f = f;
    buff->proc_map = (int *) malloc((world_size + 1) * sizeof(int));
    buff->row_map = (int *) malloc(count * sizeof(int));

    double *data = (double *) malloc(count * f * sizeof(double));
    buff->buffer = (double **) malloc(count * sizeof(double *));

    for (int i = 0; i < count; i++) {
        buff->buffer[i] = &(data[f * i]);
    }

    return buff;
}

/*
 * Communication buffer data structure added by @Kutay
 */
void CommBufferFree(CommBuffer *buff) {
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
TP_Comm *readTwoPhaseComm(char *fName, int f) {
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int64_t sloc;
    TP_Comm *Comm = (TP_Comm *) malloc(sizeof(TP_Comm));
    FILE *fpmat = fopen(fName, "rb");

    fseek(fpmat, (world_rank * sizeof(int64_t)), SEEK_SET);
    fread(&sloc, sizeof(int64_t), 1, fpmat);
    printf("id: %d, sloc: %ld\n", world_rank, sloc);

    fseek(fpmat, sloc, SEEK_SET);
    fread(&(Comm->sendBuffer_p1.count), sizeof(int), 1, fpmat);
    fread(&(Comm->recvBuffer_p1.count), sizeof(int), 1, fpmat);
    fread(&(Comm->sendBuffer_p2.count), sizeof(int), 1, fpmat);
    fread(&(Comm->recvBuffer_p2.count), sizeof(int), 1, fpmat);

    // allocate ranges
    Comm->sendBuffer_p1.proc_map = (int *) malloc((world_size + 1) * sizeof(int));
    Comm->sendBuffer_p1.row_map = (int *) malloc(Comm->sendBuffer_p1.count * sizeof(int));

    Comm->recvBuffer_p1.proc_map = (int *) malloc((world_size + 1) * sizeof(int));
    Comm->recvBuffer_p1.row_map = (int *) malloc(Comm->recvBuffer_p1.count * sizeof(int));

    Comm->sendBuffer_p2.proc_map = (int *) malloc((world_size + 1) * sizeof(int));
    Comm->sendBuffer_p2.row_map = (int *) malloc(Comm->sendBuffer_p2.count * sizeof(int));

    Comm->recvBuffer_p2.proc_map = (int *) malloc((world_size + 1) * sizeof(int));
    Comm->recvBuffer_p2.row_map = (int *) malloc(Comm->recvBuffer_p2.count * sizeof(int));

    // read proc map
    fread(Comm->sendBuffer_p1.proc_map, sizeof(int), world_size + 1, fpmat);
    fread(Comm->recvBuffer_p1.proc_map, sizeof(int), world_size + 1, fpmat);
    fread(Comm->sendBuffer_p2.proc_map, sizeof(int), world_size + 1, fpmat);
    fread(Comm->recvBuffer_p2.proc_map, sizeof(int), world_size + 1, fpmat);
    // read row map
    fread(Comm->sendBuffer_p1.row_map, sizeof(int), Comm->sendBuffer_p1.count, fpmat);
    fread(Comm->recvBuffer_p1.row_map, sizeof(int), Comm->recvBuffer_p1.count, fpmat);
    fread(Comm->sendBuffer_p2.row_map, sizeof(int), Comm->sendBuffer_p2.count, fpmat);
    fread(Comm->recvBuffer_p2.row_map, sizeof(int), Comm->recvBuffer_p2.count, fpmat);

    Comm->sendBuffer_p1.f = f;
    Comm->recvBuffer_p1.f = f;
    Comm->sendBuffer_p2.f = f;
    Comm->recvBuffer_p2.f = f;

    return Comm;
}

/*
 * parallel read of one phase communication data structure
 * Added by @Kutay
 */
OP_Comm *readOnePhaseComm(char *fName) {
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int64_t sloc;
    OP_Comm *Comm = (OP_Comm *) malloc(sizeof(OP_Comm));
    FILE *fpmat = fopen(fName, "rb");
    fseek(fpmat, (world_rank * sizeof(int64_t)), SEEK_SET);
    fread(&sloc, sizeof(int64_t), 1, fpmat);

    fseek(fpmat, sloc, SEEK_SET);
    fread(&(Comm->sendBuffer.count), sizeof(int), 1, fpmat);
    fread(&(Comm->recvBuffer.count), sizeof(int), 1, fpmat);

    Comm->sendBuffer.proc_map = (int *) malloc((world_size + 1) * sizeof(int));
    Comm->sendBuffer.row_map = (int *) malloc(Comm->sendBuffer.count * sizeof(int));

    Comm->recvBuffer.proc_map = (int *) malloc((world_size + 1) * sizeof(int));
    Comm->recvBuffer.row_map = (int *) malloc(Comm->recvBuffer.count * sizeof(int));

    fread(Comm->sendBuffer.proc_map, sizeof(int), world_size + 1, fpmat);
    fread(Comm->recvBuffer.proc_map, sizeof(int), world_size + 1, fpmat);

    fread(Comm->sendBuffer.row_map, sizeof(int), Comm->sendBuffer.count, fpmat);
    fread(Comm->recvBuffer.row_map, sizeof(int), Comm->recvBuffer.count, fpmat);

    return Comm;
}