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

    spmm(A, X, Y, comm);
    double t1, t2, t3;
    int min = 9999999;
    int i;
    for (i = 0; i < iter; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        t1 = MPI_Wtime();
        //matrix_fill_double(X, i);

        /*
		if (i%2 == 0) {
			spmm(A, X, Y, comm);
		} else {
			spmm(A, Y, X, comm);
		}
        */
        spmm(A, X, Y, comm);

        MPI_Barrier(MPI_COMM_WORLD);
        t2 = MPI_Wtime();

        if (world_rank == 0) {
            printf("runtime=> %lf\n", t2 - t1);
            //matrix_print(Y);
        }

        if (min > t2 - t1) {
            min = t2 - t1;
        }
    }
    if (world_rank == 0) {
        printf("Min runtime for current experiment=> %lf\n", min);
    }

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

    spmm(A, X, Y, comm);
    double t1, t2, t3;
    int min = 9999999;
    int i;
    for (i = 0; i < iter; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        t1 = MPI_Wtime();
        //matrix_fill_double(X, i);

        /*
		if (i%2 == 0) {
			spmm(A, X, Y, comm);
		} else {
			spmm(A, Y, X, comm);
		}
        */
        spmm(A, X, Y, comm);


        MPI_Barrier(MPI_COMM_WORLD);
        t2 = MPI_Wtime();


        if (world_rank == 0) {
            printf("runtime=> %lf\n", t2 - t1);
            //matrix_print(Y); 
        }

        if (min > t2 - t1) {
            min = t2 - t1;
        }
    }
    if (world_rank == 0) {
        printf("Min runtime for current experiment=> %lf\n", min);
    }

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
        printf("Analysis for %s, phase: %s, method: %s\n", argv[1], argv[2], argv[3]);
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
