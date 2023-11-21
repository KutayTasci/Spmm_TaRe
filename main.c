#include <stdio.h>
#include <mpi.h>
#include "inc/SparseMat.h"

int main() {
    MPI_Init(NULL, NULL);
    printf("Hello_world\n");
    MPI_Finalize();
    return 0;
}
