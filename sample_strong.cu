#include <iostream>
#include <math.h>
#include <time.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

/* if the lenght of vector is large, set this to zero */
#define PRINT_VECTOR_CONTENT 0

/* for measuring time */
double getUnixTime(void){
	struct timespec tv;
	if(clock_gettime(CLOCK_REALTIME, &tv) != 0) return 0;
	return (((double) tv.tv_sec) + (double) (tv.tv_nsec / 1000000000.0));
}

#ifdef USE_CUDA
/* CUDA stuff: */
#include "cuda.h"

/* cuda error check */ 
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"\n\x1B[31mCUDA error:\x1B[0m %s %s \x1B[33m%d\x1B[0m\n\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

/* kernel called in this example */
__global__ void mykernel(double *x_arr, int mysize, int Tstart){
	int i = blockIdx.x*blockDim.x + threadIdx.x; /* compute my id */

	if(i<mysize){ /* maybe we call more than mysize kernels */
		x_arr[i] = Tstart+i;
	}

	/* i >= mysize then relax and do nothing */	
}

/* print kernel, call only one (!)  */
/* in practical applications, this way is a real bottleneck, array should be transfered to CPU and printed there */
/* anyway, in this sample I want to print only small arrays, so who cares.. */
__global__ void printkernel(double *x_arr, int mysize){
	printf("  [");
	for(int i=0;i<mysize;i++){
		printf(" %f", x_arr[i]);
		if(i < mysize-1) printf(",");
	}
	printf(" ]\n");
}

__global__ void printkernel_id(double *x_arr, int mysize, int id, int rank){
	printf("%d_GPU: from cuda x_arr[%d]: ", rank, id);
	if(id < mysize){
		printf("%f", x_arr[id]);
	} else {
		printf("out of range");
	}
	printf("\n");
}

/* end of CUDA stuff */
#endif



int main( int argc, char *argv[] )
{
	/* load console arguments */
	int T;
	int nmb_of_tests;

	if(argc != 3){
		std::cout << "call with: ./sample_strong T nmb_of_tests" << std::endl;
		return 1;
	} else {
		T = atoi(argv[1]);
		nmb_of_tests = atoi(argv[2]);
	}

	/* MPI stuff */
    MPI_Init(NULL, NULL); /* Initialize the MPI environment */

	int MPIrank, MPIsize;
	MPI_Comm_rank(MPI_COMM_WORLD, &MPIrank);
	MPI_Comm_size(MPI_COMM_WORLD, &MPIsize);	
	
	/* compute Tlocal */
	int Tlocal_optimal = floor(T/(double)MPIsize);
	int Tlocal_optimal_rest = T - Tlocal_optimal*MPIsize;
	int Tlocal = Tlocal_optimal;
	int Tstart = Tlocal_optimal*MPIrank;
	if(MPIrank < Tlocal_optimal_rest){
		Tlocal++;
		Tstart+=MPIrank;
	} else {
		Tstart+=Tlocal_optimal_rest;
	}
	
#ifdef USE_CUDA
	/* CUDA stuff */
	gpuErrchk( cudaDeviceReset() );
#endif

	/* print problem info */
	if(MPIrank == 0){ /* only master prints */
		std::cout << "Benchmark started" << std::endl;
		std::cout << " T            = " << T << std::endl;
		std::cout << " Tlocal       = " << Tlocal << std::endl;
		std::cout << " nmb_of_tests = " << nmb_of_tests << std::endl;
		std::cout << " MPIsize      = " << MPIsize << std::endl;
	}
	MPI_Barrier( MPI_COMM_WORLD );
	std::cout << " * MPIrank  = " << MPIrank << std::endl; /* everybody say hello */
	MPI_Barrier( MPI_COMM_WORLD );
	if(MPIrank == 0){ /* only master prints */
		std::cout << std::endl;
	}
	MPI_Barrier( MPI_COMM_WORLD );
	
	double timer;
	double timer1;
	double timer2;
	
	double *x_arr; /* my array on GPU */

#ifdef USE_CUDA
/* ------- CUDA version ------- */
	int minGridSize, blockSize, gridSize; /* for optimal call */

	/* allocate array */
	timer = getUnixTime(); /* start to measure time */
	gpuErrchk( cudaMalloc(&x_arr, sizeof(double)*Tlocal) );
	
	if(MPIrank == 0){ /* only master prints */
		std::cout << " - allocation: " << getUnixTime() - timer << "s" << std::endl;
	}
		
	/* warm up */
	mykernel<<<1,1>>>(x_arr,1,1);
	gpuErrchk( cudaDeviceSynchronize() );
	MPI_Barrier( MPI_COMM_WORLD ); 	
		
	/* compute optimal parameters of the call */
	gpuErrchk( cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize,mykernel, 0, 0) );
	gridSize = (Tlocal + blockSize - 1)/ blockSize;

	timer = getUnixTime();
			
	for(int k=0;k<nmb_of_tests;k++){
		mykernel<<<gridSize, blockSize>>>(x_arr, Tlocal, Tstart);
		gpuErrchk( cudaDeviceSynchronize() );
		MPI_Barrier( MPI_COMM_WORLD ); 
	}

	printkernel_id<<<1,1>>>(x_arr, Tlocal, 0, MPIrank);
	gpuErrchk( cudaDeviceSynchronize() );
	printkernel_id<<<1,1>>>(x_arr, Tlocal, 1000, MPIrank);
	gpuErrchk( cudaDeviceSynchronize() );
	printkernel_id<<<1,1>>>(x_arr, Tlocal, Tlocal-1, MPIrank);
	gpuErrchk( cudaDeviceSynchronize() );

	timer1 = getUnixTime() - timer;
	for(int k=0;k<MPIsize;k++){
		if(k==MPIrank){
			/* my turn - I am printing */
			std::cout << MPIrank << ". ( gridSize = " << gridSize << ", blockSize = " << blockSize << " )" << std::endl;
		}
		MPI_Barrier( MPI_COMM_WORLD );
	}

	/* print array */
	if(PRINT_VECTOR_CONTENT){
		timer = getUnixTime();

		for(int k=0;k<MPIsize;k++){
			if(k==MPIrank){
				/* my turn - I am printing */
				std::cout << k << ".CPU:" << std::endl;

				printkernel<<<1,1>>>(x_arr,Tlocal);
				gpuErrchk( cudaDeviceSynchronize() );
			}
			
			MPI_Barrier( MPI_COMM_WORLD );
		}


		if(MPIrank == 0){ /* only master prints */
			std::cout << " - printed in: " << getUnixTime() - timer << "s" << std::endl;
		}
		MPI_Barrier( MPI_COMM_WORLD );

	}

	/* destroy array */
	timer = getUnixTime();
	gpuErrchk( cudaFree(x_arr) );
	MPI_Barrier( MPI_COMM_WORLD );

	if(MPIrank == 0){ /* only master prints */
		std::cout << " - destruction: " << getUnixTime() - timer << "s" << std::endl;
	}
	MPI_Barrier( MPI_COMM_WORLD );

#else
/* ------- SEQUENTIAL version ------- */

	/* allocate array */
	timer = getUnixTime();
	x_arr = new double[Tlocal];
	MPI_Barrier( MPI_COMM_WORLD );

	if(MPIrank == 0){ /* only master prints */
		std::cout << " - allocation: " << getUnixTime() - timer << "s" << std::endl;
	}
	MPI_Barrier( MPI_COMM_WORLD );

	/* fill array */
	timer = getUnixTime();
	for(int k=0;k<nmb_of_tests;k++){
		for(int i=0;i<Tlocal;i++){
			x_arr[i] = Tstart+i;
		}
		MPI_Barrier( MPI_COMM_WORLD );
	}
	timer2 = getUnixTime() - timer;
	MPI_Barrier( MPI_COMM_WORLD );
		
	/* print array */
	if(PRINT_VECTOR_CONTENT){
		timer = getUnixTime();

		for(int k=0;k<MPIsize;k++){
			if(k==MPIrank){
				/* my turn - I am printing */
				std::cout << k << ".CPU:" << std::endl;

				std::cout << "  [";
				for(int i=0;i<Tlocal;i++){
					std::cout << " " << x_arr[i];
					if(i < Tlocal-1) std::cout << ",";
				}
				std::cout << " ]" << std::endl;
			}
			
			MPI_Barrier( MPI_COMM_WORLD );
		}

		if(MPIrank == 0){ /* only master prints */
			std::cout << " - printed in: " << getUnixTime() - timer << "s" << std::endl;
		}
		MPI_Barrier( MPI_COMM_WORLD );
	}
	MPI_Barrier( MPI_COMM_WORLD );
		
	/* destroy array */
	timer = getUnixTime();
	delete [] x_arr;
	MPI_Barrier( MPI_COMM_WORLD );

	if(MPIrank == 0){ /* only master prints */
		std::cout << " - destruction: " << getUnixTime() - timer << "s" << std::endl;
	}
	MPI_Barrier( MPI_COMM_WORLD );
	
#endif


	/* final print of timers */
	if(MPIrank==0){ /* only master prints */
		std::cout << std::endl;
		std::cout << "---- TIMERS ----" << std::endl;
#ifdef USE_CUDA
		std::cout << " GPU = " << timer1 << "s" << std::endl;
#else
		std::cout << " CPU = " << timer2 << "s" << std::endl;
#endif
		std::cout << std::endl;
	}
	MPI_Barrier( MPI_COMM_WORLD );
	
	/* MPI stuff */
	MPI_Finalize();	/* Finalize the MPI environment. */

	return 0;
}


