#include <mpi.h>
#include <iostream>

int main () {

    int provided;
	MPI_Init_thread(0, 0, MPI_THREAD_FUNNELED, &provided);

    int myRank, worldSize;
	MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    std::cout << "Hello from rank " << myRank << std::endl;

    MPI_Finalize();

    return 0;
}