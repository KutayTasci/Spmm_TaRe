//
// Created by kutay on 20.11.2023.
//
#include "../inc/CommHandler.h"
#include "../inc/DenseMat.h"
#include <stdlib.h>
#include <mpi.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>


void shuffle(int array[], int n) {
    srand(time(NULL)); // Seed the random number generator
    for (int i = n - 1; i > 0; i--) {
        // Generate a random index j such that 0 <= j <= i
        int j = rand() % (i + 1);

        // Swap array[i] and array[j]
        int temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

/*
 * Reading CSR Matrix in parallel
 * Handles memory allocation
 * File format follows the output format of "Util" code by Oguz Selvitopi
 * Added by @Kutay
*/

/*
 * Communication buffer data structure added by @Kutay
 */
void CommBufferInit(CommBuffer *buff) {
    double *data = (double *) malloc(buff->count * buff->f * sizeof(double));
    buff->buffer = (double **) malloc(buff->count * sizeof(double *));

    for (int i = 0; i < buff->count; i++) {
        buff->buffer[i] = &(data[buff->f * i]);
    }

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
TP_Comm *readTwoPhaseComm(char *fName, int f, bool partial_reduce) {
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int64_t sloc;
    TP_Comm *Comm = (TP_Comm *) malloc(sizeof(TP_Comm));
    FILE *fpmat = fopen(fName, "rb");
    if (fpmat == NULL) {
        if (world_rank == 0)
            printf("Phase comm file named '%s' not found\n", fName);
        exit(1);
    }
    fseek(fpmat, (world_rank * sizeof(int64_t)), SEEK_SET);
    fread(&sloc, sizeof(int64_t), 1, fpmat);


    fseek(fpmat, sloc, SEEK_SET);
    fread(&(Comm->sendBuffer_p1.count), sizeof(int), 1, fpmat);
    fread(&(Comm->recvBuffer_p1.count), sizeof(int), 1, fpmat);
    fread(&(Comm->sendBuffer_p2.count), sizeof(int), 1, fpmat);
    fread(&(Comm->recvBuffer_p2.count), sizeof(int), 1, fpmat);


    Comm->sendBuffer_p1.proc_map = (int *) malloc((world_size + 1) * sizeof(int));
    Comm->sendBuffer_p1.row_map = (int *) malloc(Comm->sendBuffer_p1.count * sizeof(int));

    Comm->recvBuffer_p1.proc_map = (int *) malloc((world_size + 1) * sizeof(int));
    Comm->recvBuffer_p1.row_map = (int *) malloc(Comm->recvBuffer_p1.count * sizeof(int));

    Comm->sendBuffer_p2.proc_map = (int *) malloc((world_size + 1) * sizeof(int));
    Comm->sendBuffer_p2.row_map = (int *) malloc(Comm->sendBuffer_p2.count * sizeof(int));

    Comm->recvBuffer_p2.proc_map = (int *) malloc((world_size + 1) * sizeof(int));
    Comm->recvBuffer_p2.row_map = (int *) malloc(Comm->recvBuffer_p2.count * sizeof(int));

    fread(Comm->sendBuffer_p1.proc_map, sizeof(int), world_size + 1, fpmat);
    fread(Comm->recvBuffer_p1.proc_map, sizeof(int), world_size + 1, fpmat);
    fread(Comm->sendBuffer_p2.proc_map, sizeof(int), world_size + 1, fpmat);
    fread(Comm->recvBuffer_p2.proc_map, sizeof(int), world_size + 1, fpmat);
    Comm->msgRecvCount_p1 = 0;
    Comm->msgRecvCount_p2 = 0;
    Comm->msgSendCount_p1 = 0;
    Comm->msgSendCount_p2 = 0;

    for (int i = 1; i <= world_size; ++i) {
        if (Comm->sendBuffer_p1.proc_map[i] - Comm->sendBuffer_p1.proc_map[i - 1] != 0) {
            Comm->msgSendCount_p1++;
        }
        if (Comm->sendBuffer_p2.proc_map[i] - Comm->sendBuffer_p2.proc_map[i - 1] != 0) {
            Comm->msgSendCount_p2++;
        }
        if (Comm->recvBuffer_p1.proc_map[i] - Comm->recvBuffer_p1.proc_map[i - 1] != 0) {
            Comm->msgRecvCount_p1++;
        }
        if (Comm->recvBuffer_p2.proc_map[i] - Comm->recvBuffer_p2.proc_map[i - 1] != 0) {
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

    if (partial_reduce != 0) {
        Comm->reducer.init = true;
        fread(&(Comm->reducer.reduce_count), sizeof(int), 1, fpmat);
        Comm->reducer.reduce_list = (int *) malloc(Comm->reducer.reduce_count * sizeof(int));
        Comm->reducer.reduce_list_mapped = (int *) malloc(Comm->reducer.reduce_count * sizeof(int));
        Comm->reducer.reduce_source_mapped = (int **) malloc(Comm->reducer.reduce_count * sizeof(int *));
        Comm->reducer.reduce_factors = (double **) malloc(Comm->reducer.reduce_count * sizeof(double *));

        for (int i = 0; i < Comm->reducer.reduce_count; i++) {
            fread(&(Comm->reducer.reduce_list[i]), sizeof(unsigned int), 1, fpmat);
            int tmp;
            fread(&(tmp), sizeof(int), 1, fpmat);
            Comm->reducer.reduce_source_mapped[i] = (int *) malloc((tmp + 1) * sizeof(int));
            Comm->reducer.reduce_factors[i] = (double *) malloc(tmp * sizeof(double));
            Comm->reducer.reduce_source_mapped[i][0] = tmp;
            fread(&(Comm->reducer.reduce_source_mapped[i][1]), sizeof(int), tmp, fpmat);
            fread(Comm->reducer.reduce_factors[i], sizeof(double), tmp, fpmat);
        }
    } else {
        Comm->reducer.init = false;
    }

    return Comm;
}


void prep_comm_tp(TP_Comm *Comm) {
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int i;
    int base, range, part;
    int *send_ls1 = (int *) malloc(Comm->msgSendCount_p1 * sizeof(int));
    int *recv_ls1 = (int *) malloc(Comm->msgRecvCount_p1 * sizeof(int));
    int *send_ls2 = (int *) malloc(Comm->msgSendCount_p2 * sizeof(int));
    int *recv_ls2 = (int *) malloc(Comm->msgRecvCount_p2 * sizeof(int));

    int ctr = 0, ctrp = 0;
    for (int i = 0; i < world_size; ++i) {
        if (Comm->sendBuffer_p1.proc_map[i + 1] - Comm->sendBuffer_p1.proc_map[i] != 0) {
            send_ls1[ctr] = i;
            ctr++;
        }


        if (Comm->recvBuffer_p1.proc_map[i + 1] - Comm->recvBuffer_p1.proc_map[i] != 0) {
            recv_ls1[ctrp] = i;
            ctrp++;
        }
    }
    ctr = 0, ctrp = 0;
    for (int i = 0; i < world_size; ++i) {
        if (Comm->sendBuffer_p2.proc_map[i + 1] - Comm->sendBuffer_p2.proc_map[i] != 0) {
            send_ls2[ctr] = i;
            ctr++;
        }


        if (Comm->recvBuffer_p2.proc_map[i + 1] - Comm->recvBuffer_p2.proc_map[i] != 0) {
            recv_ls2[ctrp] = i;
            ctrp++;
        }
    }

    shuffle(send_ls1, Comm->msgSendCount_p1);
    Comm->send_proc_list_p1 = send_ls1;
    Comm->recv_proc_list_p1 = recv_ls1;
    shuffle(send_ls2, Comm->msgSendCount_p2);
    Comm->send_proc_list_p2 = send_ls2;
    Comm->recv_proc_list_p2 = recv_ls2;

    Comm->send_ls_p1 = (MPI_Request *) malloc((Comm->msgSendCount_p1) * sizeof(MPI_Request));
    Comm->recv_ls_p1 = (MPI_Request *) malloc((Comm->msgRecvCount_p1) * sizeof(MPI_Request));
    Comm->send_ls_p2 = (MPI_Request *) malloc((Comm->msgSendCount_p2) * sizeof(MPI_Request));
    Comm->recv_ls_p2 = (MPI_Request *) malloc((Comm->msgRecvCount_p2) * sizeof(MPI_Request));
}


/*
 * parallel read of one phase communication data structure
 * Added by @Kutay
 */
OP_Comm *readOnePhaseComm(char *fName, int f) {
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

    Comm->msgRecvCount = 0;
    Comm->msgSendCount = 0;

    for (int i = 1; i <= world_size; ++i) {
        if (Comm->sendBuffer.proc_map[i] - Comm->sendBuffer.proc_map[i - 1] != 0) {
            Comm->msgSendCount++;
        }
        if (Comm->recvBuffer.proc_map[i] - Comm->recvBuffer.proc_map[i - 1] != 0) {
            Comm->msgRecvCount++;
        }
    }


    Comm->sendBuffer.f = f;
    Comm->recvBuffer.f = f;

    CommBufferInit(&(Comm->sendBuffer));


    return Comm;
}

void prep_comm_op(OP_Comm *Comm) {
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int *send_ls = (int *) malloc(Comm->msgSendCount * sizeof(int));
    int *recv_ls = (int *) malloc(Comm->msgRecvCount * sizeof(int));

    int ctr = 0, ctrp = 0;
    for (int i = 0; i < world_size; ++i) {
        if (Comm->sendBuffer.proc_map[i + 1] - Comm->sendBuffer.proc_map[i] != 0) {
            send_ls[ctr] = i;
            ctr++;
        }
        if (Comm->recvBuffer.proc_map[i + 1] - Comm->recvBuffer.proc_map[i] != 0) {
            recv_ls[ctrp] = i;
            ctrp++;
        }
    }

    Comm->send_ls = (MPI_Request *) malloc((Comm->msgSendCount) * sizeof(MPI_Request));
    Comm->recv_ls = (MPI_Request *) malloc((Comm->msgRecvCount) * sizeof(MPI_Request));

    shuffle(send_ls, Comm->msgSendCount);
    Comm->send_proc_list = send_ls;
    Comm->recv_proc_list = recv_ls;
}
