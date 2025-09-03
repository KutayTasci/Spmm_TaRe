#include <assert.h>
#include <stdio.h>
#include <mpi.h>
#include <string.h>
#include <unistd.h>
#include <sys/resource.h>

#include "inc/SparseMat.h"
#include "inc/CommHandler.h"
#include "inc/DenseMat.h"
#include "inc/SpMM.h"
#include "inc/Reader.h"

void matrix_print(Matrix* m) {
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

long get_memory_usage() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss; // Memory usage in KB
}


MPI_Info create_tuned_info() {
    MPI_Info info;
    MPI_Info_create(&info);

    // These hints are implementation-dependent, but often safe and useful
    MPI_Info_set(info, "no_any_source", "true"); // no MPI_ANY_SOURCE
    MPI_Info_set(info, "no_any_tag", "true"); // no MPI_ANY_TAG
    MPI_Info_set(info, "mpi_assert_exact_length", "true"); // message sizes will match exactly

    return info;
}

void calculate_and_print_runtimes(float* runtimes, int iter, int world_rank) {
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
        printf("%.6f,%.6f,%.6f", min, max, avg);
        // deallocate the runtimes array
        free(runtimes);
    }
}


void test_op(ReaderRet* args, void (*spmm)(SparseMat*, Matrix*, Matrix*, OP_Comm*, int, wct*)) {
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    MPI_Barrier(MPI_COMM_WORLD);
    SparseMat* A = readSparseMat(args->f_mat, STORE_BY_ROWS, args->f_inpart);
    MPI_Barrier(MPI_COMM_WORLD);
    OP_Comm* comm = readOnePhaseComm(args->f_comm, args->k, args->reduce);
    Matrix* X = matrix_create_op(A->m, args->k, A->gn, args->k, comm);
    matrix_fill_double(X, 0.0);
    Matrix* Y = matrix_create_op(A->m, args->k, A->gn, args->k, comm);
    map_csr_op(A, comm);
    prep_comm_op(comm);
    if (args->op_method == P2P) {
        map_comm_op(comm, X);
    }
    else if (args->op_method == ALL2ALLV) {
        map_comm_all2allv(comm, X);
    }
    else if (args->op_method == NEGHB_ALL2ALLV || args->op_method == NEGHB_ALL2ALLV_REORDER) {
        MPI_Dist_graph_create_adjacent(MPI_COMM_WORLD, comm->msgRecvCount, comm->recv_proc_list, MPI_UNWEIGHTED,
                                       comm->msgSendCount, comm->send_proc_list, MPI_UNWEIGHTED,
                                       MPI_INFO_NULL, args->op_method == NEGHB_ALL2ALLV_REORDER,
                                       &comm->custom_comm);
        map_comm_nall2allv(comm, X);
    }
    float* runtimes = (float*)malloc(args->iter * sizeof(float));
    int i;
    wct times = wct_init();

    //warmup iteration
    for (i = 0; i < 3; i++) {
        spmm(A, X, Y, comm, WCT_FULL, &times);
    }


    for (i = 0; i < args->iter; i++) {
        matrix_fill_double(X, 0.0);


        spmm(A, X, Y, comm, WCT_FULL, &times);


        if (world_rank == 0) {
            runtimes[i] = times.total_t;
        }
    }
    calculate_and_print_runtimes(runtimes, args->iter, world_rank);
    times.total_t = 0; //reset the total time
    spmm(A, X, Y, comm, WCT_PROFILE, &times);
    if (world_rank == 0) {
        wct_print(&times);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    matrix_free(X);
    sparseMatFree(A);
}

void test_tp(ReaderRet* args, void (*spmm)(SparseMat*, Matrix*, Matrix*, TP_Comm*, int, wct*)) {
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    SparseMat* A = readSparseMat(args->f_mat, STORE_BY_ROWS, args->f_inpart);
    //FOR PARTIAL REDUCE TP_PARTIAL_REDUCE OR TP_STANDARD FOR NOR REDUCE
    TP_Comm* comm = readTwoPhaseComm(args->f_comm, args->k, args->reduce);
    Matrix* X = matrix_create_tp(A->m, args->k, A->gn, args->k, comm);

    matrix_fill_double(X, 1.0);
    Matrix* Y = matrix_create_tp(A->m, args->k, A->gn, args->k, comm);

    map_csr(A, comm);

    prep_comm_tp(comm);
    map_comm_tp(comm, X);
    if (world_rank == 0) {
        fflush(stdout);
    }

    int i;
    wct times = wct_init();
    // 5 iteration warmup
    for (i = 0; i < 5; i++) {
        spmm(A, X, Y, comm, WCT_FULL, &times);
    }
    float* runtimes = (float*)malloc(args->iter * sizeof(float));

    for (i = 0; i < args->iter; i++) {
        matrix_fill_double(X, 0.0);

        spmm(A, X, Y, comm, WCT_FULL, &times);


        if (world_rank == 0) {
            runtimes[i] = times.total_t;
        }
    }

    calculate_and_print_runtimes(runtimes, args->iter, world_rank);
    times.total_t = 0; //reset the total time
    spmm(A, X, Y, comm, WCT_PROFILE, &times);
    if (world_rank == 0) {
        wct_print(&times);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    matrix_free(X);
    sparseMatFree(A);
}


int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    ReaderRet parsedArgs = parseFileFromArgs(argc, argv);
    if (!parsedArgs.is_valid) {
        MPI_Finalize();
        return 1;
    }

    if (world_rank == 0) {
        char* dataset_name = strrchr(argv[1], '/');
        dataset_name++; // skip "/"
        // the csv headers are: dataset_name,comm_type,spmm_type,min_runtime,max_runtime,avg_runtime
        // runtime fields will be filled in the test functions
        printf("%s,%s,%s,", dataset_name, argv[2], argv[3]);
    }

    if (parsedArgs.one_phase) {
        switch (parsedArgs.op_method) {
        case P2P:
            test_op(&parsedArgs, &spmm_op);
            break;
        case ALL2ALLV:
            test_op(&parsedArgs, &spmm_alltoallv);
            break;
        case NEGHB_ALL2ALLV:
        case NEGHB_ALL2ALLV_REORDER:
            test_op(&parsedArgs, &spmm_alltoallv_neghb);
            break;
        default:
            if (world_rank == 0) {
                printf("Invalid operation method: %d\n", parsedArgs.op_method);
            }
        }
    }
    else {
        test_tp(&parsedArgs, &spmm_tp);
    }
    MPI_Finalize();
    return 0;
}
