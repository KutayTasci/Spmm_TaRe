cmake_minimum_required(VERSION 3.15.4)
project(Spmm_TaRe C)

set(CMAKE_C_COMPILER icx)
set(CMAKE_CXX_COMPILER icpx)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(MPI REQUIRED)
find_package(MKL REQUIRED)

if (BLOCKING_COMM)
    message(STATUS "Blocking communication enabled")
    add_definitions(-DBLOCKING_COMM)
endif ()
if (USE_RSEND)
    message(STATUS "Using ready send for communication")
    add_definitions(-DUSE_RSEND)
endif ()
if (USE_I64)
    message(STATUS "Using 64-bit integers (for row pointers)")
    add_definitions(-Duse_i64)
endif (USE_I64)

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

target_link_libraries(Spmm_TaRe PUBLIC MPI::MPI_C MKL::MKL)
