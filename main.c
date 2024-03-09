#include <stdio.h>
#include <mpi.h>
#include <string.h>
#include "inc/SparseMat.h"
#include "inc/CommHandler.h"
#include "inc/DenseMat.h"
#include "inc/SpMM.h"

void matrix_print(Matrix* m) {
    printf("Rows: %d Columns: %d\n", m->m, m->n);
    for (int i = 0; i < 1; i++) {
        for (int j = 0; j < m->n; j++) {
            printf("%1.7f ", m->entries[i][j]);
        }
        printf("\n");
    }
}

void test_op(char* f_inpart, char* f_mat, char* f_comm, int k, int iter, void (*spmm)()) {
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
	for (int i=0;i<iter;i++) {
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

void test_tp(char* f_inpart, char* f_mat, char* f_comm, int k, int iter, void (*spmm)()) {
	
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
	for (int i=0;i<iter;i++) {
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
    
    char f_inpart[100];
    char f_mat[100];
    char f_comm[100];
    
    strcpy(f_inpart, argv[1]);
    strcpy(f_mat, argv[1]);
    strcpy(f_comm, argv[1]);
    
    char partf[100];
    
    sprintf(partf, "/adj.inpart.%d",world_size);
    strcat(f_inpart, partf);
    sprintf(partf, "/adj.inpart.%d.bin",world_size);
    strcat(f_mat, partf);
    
    //char* f_inpart = "/home/kutay/TaskReassignment/data/Yelp/Yel.inpart.64";
    //char* f_mat = "/home/kutay/TaskReassignment/data/Yelp/Yel.inpart.64.bin";
    //char* f_comm = "/home/kutay/TaskReassignment/data/Yelp/Yel.phases.64.bin";
	
	
	
    if (strcmp(argv[2], "op") == 0) {
    	sprintf(partf, "/adj.phases.%d.one.bin",world_size);
    	strcat(f_comm, partf);
    	
    	if (world_rank == 0) {
			printf("%s \n", f_inpart);
			printf("%s \n", f_mat);
			printf("%s \n", f_comm);
		}
			
    	if (strcmp(argv[3], "reduce") == 0) {
    		test_op(f_inpart, f_mat, f_comm, atoi(argv[4]), atoi(argv[5]), &spmm_reduce_op);
    	} else {
    		test_op(f_inpart, f_mat, f_comm, atoi(argv[4]), atoi(argv[5]), &spmm_op);
    	}
    	
    	
    } else {
    	sprintf(partf, "/adj.phases.%d.bin",world_size);
    	strcat(f_comm, partf);
    	if (strcmp(argv[3], "reduce") == 0) {
    		test_tp(f_inpart, f_mat, f_comm, atoi(argv[4]), atoi(argv[5]), &spmm_reduce_tp);
    	} else {
    		test_tp(f_inpart, f_mat, f_comm, atoi(argv[4]), atoi(argv[5]), &spmm_tp);
    	}
    	
    }
    MPI_Finalize();
    return 0;
}
