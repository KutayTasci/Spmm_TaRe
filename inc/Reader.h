//
// Created by serdar on 6/29/24.
//

#ifndef SPMM_TARE_READER_H
#define SPMM_TARE_READER_H

#include <stdio.h>
#include <mpi.h>
#include <string.h>
#include <stdbool.h>
#include <dirent.h>
#include <stdlib.h>

enum OP_METHOD {
    P2P = 0,
    ALL2ALLV = 1,
    NEGHB_ALL2ALLV = 2,
    NEGHB_ALL2ALLV_REORDER = 3,
};

typedef struct ReaderRet {
    bool is_valid;
    bool one_phase;
    bool reduce;
    enum OP_METHOD op_method;
    int k;
    int iter;
    char f_inpart[200];
    char f_mat[200];
    char f_comm[200]; // 2phase or 1phase
    char dataset_name[100];
} ReaderRet;

ReaderRet parseFileFromArgs(int argc, char* argv[]);

#endif //SPMM_TARE_READER_H
