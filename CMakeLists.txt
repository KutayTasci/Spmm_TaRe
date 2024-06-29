cmake_minimum_required(VERSION 3.15.4)
project(Spmm_TaRe C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(MPI REQUIRED)

add_executable(Spmm_TaRe main.c
        inc/SparseMat.h
        matrix/SparseMat.c
        inc/DenseMat.h
        matrix/DenseMat.c
        src/CommHandler.c
        inc/CommHandler.h
        inc/SpMM.h
        matrix/SpMM.c
        inc/Reader.h
        src/Reader.c)

target_link_libraries(Spmm_TaRe PUBLIC MPI::MPI_C)
