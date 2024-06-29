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

typedef struct ReaderRet {
    bool is_valid;
    bool one_phase;
    bool reduce;
    int k;
    int iter;
    char f_inpart[100];
    char f_mat[100];
    char f_comm[100]; // 2phase or 1phase
    char dataset_name[50];
} ReaderRet;

ReaderRet parseFileFromArgs(int argc, char *argv[]);

#endif //SPMM_TARE_READER_H
