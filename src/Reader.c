//
// Created by serdar on 6/29/24.
//

#include "../inc/Reader.h"

ReaderRet parseFileFromArgs(int argc, char *argv[]) {
    ReaderRet ret;
    ret.is_valid = false;

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc != 6) {
        if (world_rank == 0) {
            printf("Usage: %s <directory> <op|tp> <reduce|noreduce> <k> <iter>\n", argv[0]);
        }
        return ret;
    }
    ret.one_phase = strstr(argv[2], "op") != NULL;
    ret.reduce = strstr(argv[3], "noreduce") == NULL;
    if (ret.one_phase && ret.reduce) {
        printf("One phase does not support reduce\n");
        return ret;
    }
    // strings to compare
    char inpart_str[100];
    char mat_str[100];
    char comm_str[100];
    if (ret.reduce) {
        sprintf(inpart_str, "inpart.reduced.%d", world_size);
        sprintf(mat_str, "inpart.reduced.%d.bin", world_size);
    } else {
        sprintf(inpart_str, "inpart.%d", world_size);
        sprintf(mat_str, "inpart.%d.bin", world_size);
    }
    if (ret.one_phase) {
        sprintf(comm_str, "phases.%d.one.bin", world_size);
    } else {
        if (ret.reduce) {
            sprintf(comm_str, "phases.%d.reduced.bin", world_size);
        } else {
            sprintf(comm_str, "phases.%d.noreduce.bin", world_size);
        }
    }
    // open the directory
    DIR *d;
    struct dirent *dir;
    d = opendir(argv[1]);
    if (d) {
        while ((dir = readdir(d)) != NULL) {
            // mat should come first since it contains inpart
            if (strstr(dir->d_name, mat_str) != NULL) {
                strcpy(ret.f_mat, argv[1]);
                strcat(ret.f_mat, "/");
                strcat(ret.f_mat, dir->d_name);
            } else if (strstr(dir->d_name, inpart_str) != NULL) {
                strcpy(ret.f_inpart, argv[1]);
                strcat(ret.f_inpart, "/");
                strcat(ret.f_inpart, dir->d_name);
            } else if (strstr(dir->d_name, comm_str) != NULL) {
                strcpy(ret.f_comm, argv[1]);
                strcat(ret.f_comm, "/");
                strcat(ret.f_comm, dir->d_name);
            }
        }
        closedir(d);
    } else {
        printf("Directory %s not found\n", argv[1]);
        return ret;
    }
    char *dataset_name = strrchr(argv[1], '/');
    dataset_name++; // skip "/"
    strcpy(ret.dataset_name, dataset_name);
    // check if the files are found
    if (strlen(ret.f_comm) == 0 || strlen(ret.f_inpart) == 0 || strlen(ret.f_mat) == 0) {
        printf("Files not found\n");
        return ret;
    }
    ret.k = atoi(argv[4]);
    ret.iter = atoi(argv[5]);

    ret.is_valid = true;
    return ret;
}