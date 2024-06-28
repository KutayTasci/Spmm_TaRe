#include <stdio.h>
#include <mpi.h>
#include <string.h>
#include "inc/SparseMat.h"
#include "inc/CommHandler.h"
#include "inc/DenseMat.h"
#include "inc/SpMM.h"
#include <dirent.h>

void matrix_print(Matrix *m) {
    printf("Rows: %d Columns: %d\n", m->m, m->n);
    int i;
    for (i = 0; i < 1; i++) {
        int j;
        for (j = 0; j < m->n; j++) {
            printf("%1.7f ", m->entries[i][j]);
        }
        printf("\n");
    }
}

void calculate_and_print_runtimes(float *runtimes, int iter, int world_rank) {
    if (world_rank == 0) {
        float min = runtimes[0], max = runtimes[0], avg = runtimes[0];
        for (int i = 1; i < iter; i++) {
            if (runtimes[i] < min) {
                min = runtimes[i];
            }
            if (runtimes[i] > max) {
                max = runtimes[i];
            }
            avg += runtimes[i];
        }
        avg = avg / iter;
        printf("%.6f,%.6f,%.6f\n", min, max, avg);
        // deallocate the runtimes array
        free(runtimes);
    }
}

void test_op(char *f_inpart, char *f_mat, char *f_comm, int k, int iter, void (*spmm)()) {
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    MPI_Barrier(MPI_COMM_WORLD);
    SparseMat *A = readSparseMat(f_mat, STORE_BY_ROWS, f_inpart);
    MPI_Barrier(MPI_COMM_WORLD);
    OP_Comm *comm = readOnePhaseComm(f_comm, k);
    Matrix *X = matrix_create_op(A->m, k, A->gn, k, comm);
    matrix_fill_double(X, 0.0);
    Matrix *Y = matrix_create_op(A->m, k, A->gn, k, comm);

    map_csr_op(A, comm);
    prep_comm_op(comm);
    map_comm_op (comm, X);
    float *runtimes = (float *) malloc(iter * sizeof(float));
    int i;
    for (i = 0;i < 10; i++) {
		spmm(A, X, Y, comm);
	}
    double t1, t2, t3;

    
    for (i = 0; i < iter; i++) {
    	matrix_fill_double(X, 0.0);
        MPI_Barrier(MPI_COMM_WORLD);
        t1 = MPI_Wtime();

        spmm(A, X, Y, comm);

        MPI_Barrier(MPI_COMM_WORLD);
        t2 = MPI_Wtime();

        if (world_rank == 0) {
            runtimes[i] = t2 - t1;
        }

    }
    calculate_and_print_runtimes(runtimes, iter, world_rank);


    MPI_Barrier(MPI_COMM_WORLD);
    matrix_free(X);
    sparseMatFree(A);
}

void test_tp(char *f_inpart, char *f_mat, char *f_comm, int k, int iter, void (*spmm)()) {

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    SparseMat *A = readSparseMat(f_mat, STORE_BY_ROWS, f_inpart);
    TP_Comm *comm = readTwoPhaseComm(f_comm, k);
    Matrix *X = matrix_create_tp(A->m, k, A->gn, k, comm);
    matrix_fill_double(X, 1.0);
    Matrix *Y = matrix_create_tp(A->m, k, A->gn, k, comm);

    map_csr(A, comm);
    prep_comm_tp(comm);
	map_comm_tp (comm, X);
	int i;
	for (i = 0;i < 10; i++) {
		spmm(A, X, Y, comm);
	}
    
    float *runtimes = (float *) malloc(iter * sizeof(float));
    double t1, t2, t3;
    for (i = 0; i < iter; i++) {
    	matrix_fill_double(X, 0.0);
        MPI_Barrier(MPI_COMM_WORLD);
        t1 = MPI_Wtime();

        spmm(A, X, Y, comm);


        MPI_Barrier(MPI_COMM_WORLD);
        t2 = MPI_Wtime();


        if (world_rank == 0) {
            runtimes[i] = t2 - t1;
        }

    }
    calculate_and_print_runtimes(runtimes, iter, world_rank);

    MPI_Barrier(MPI_COMM_WORLD);
    matrix_free(X);
    sparseMatFree(A);

}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // file names
    char f_inpart[100];
    char f_mat[100];
    char f_comm[100];
    char f_one_comm[100];
    // strings to compare
    char inpart_str[100];
    char mat_str[100];
    char comm_str[100];
    char one_comm_str[100];
    sprintf(inpart_str, "inpart.%d", world_size);
    sprintf(mat_str, "inpart.%d.bin", world_size);
    sprintf(comm_str, "phases.%d.bin", world_size);
    sprintf(one_comm_str, "phases.%d.one.bin", world_size);
    // open the directory
    DIR *d;
    struct dirent *dir;
    d = opendir(argv[1]);
    if (d) {
        while ((dir = readdir(d)) != NULL) {
            // mat should come first since it contains inpart
            if (strstr(dir->d_name, mat_str) != NULL) {
                strcpy(f_mat, argv[1]);
                strcat(f_mat, "/");
                strcat(f_mat, dir->d_name);
            } else if (strstr(dir->d_name, inpart_str) != NULL) {
                strcpy(f_inpart, argv[1]);
                strcat(f_inpart, "/");
                strcat(f_inpart, dir->d_name);
            } else if (strstr(dir->d_name, comm_str) != NULL) {
                strcpy(f_comm, argv[1]);
                strcat(f_comm, "/");
                strcat(f_comm, dir->d_name);
            } else if (strstr(dir->d_name, one_comm_str) != NULL) {
                strcpy(f_one_comm, argv[1]);
                strcat(f_one_comm, "/");
                strcat(f_one_comm, dir->d_name);
            }
        }
        closedir(d);
    } else {
        printf("Directory %s not found\n", argv[1]);
        return 1;
    }
    if (world_rank == 0) {
        char *dataset_name = strrchr(argv[1], '/');
        dataset_name++; // skip "/"
        // the csv headers are: dataset_name,comm_type,spmm_type,min_runtime,max_runtime,avg_runtime
        // runtime fields will be filled in the test functions
        printf("%s,%s,%s,", dataset_name, argv[2], argv[3]);
    }

    if (strcmp(argv[2], "op") == 0) {

        if (strcmp(argv[3], "reduce") == 0) {
            test_op(f_inpart, f_mat, f_one_comm, atoi(argv[4]), atoi(argv[5]), &spmm_reduce_op);
        } else {
            test_op(f_inpart, f_mat, f_one_comm, atoi(argv[4]), atoi(argv[5]), &spmm_op);
        }


    } else {
        if (strcmp(argv[3], "reduce") == 0) {
            test_tp(f_inpart, f_mat, f_comm, atoi(argv[4]), atoi(argv[5]), &spmm_reduce_tp);
        } else {
            test_tp(f_inpart, f_mat, f_comm, atoi(argv[4]), atoi(argv[5]), &spmm_tp);
        }

    }
    MPI_Finalize();
    return 0;
}
