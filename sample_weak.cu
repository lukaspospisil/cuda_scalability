#include <iostream>
#include <math.h>
#include <time.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

/* if the lenght of vector is large, set this to zero */
#define PRINT_VECTOR_CONTENT 0

/* which CUDA calls to test? */
#define CALL_NAIVE 1
#define CALL_OPTIMAL 1

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


/* end of CUDA stuff */
#endif



int main( int argc, char *argv[] )
{
	/* load console arguments */
	int Tlocal;
	int nmb_of_tests;

	if(argc != 3){
		std::cout << "call with: ./sample_weak Tlocal nmb_of_tests" << std::endl;
		return 1;
	} else {
		Tlocal = atoi(argv[1]);
		nmb_of_tests = atoi(argv[2]);
	}

	/* MPI stuff */
    MPI_Init(NULL, NULL); /* Initialize the MPI environment */

	int MPIrank, MPIsize;
	MPI_Comm_rank(MPI_COMM_WORLD, &MPIrank);
	MPI_Comm_size(MPI_COMM_WORLD, &MPIsize);	
	
#ifdef USE_CUDA
	/* CUDA stuff */
	gpuErrchk( cudaDeviceReset() );
#endif

	int T = MPIsize*Tlocal; 

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
	double timer3;
	
	double *x_arr; /* my array on GPU */
	int Tstart; /* lower range of local part */

	/* compute local length of array */
	Tstart = MPIrank*Tlocal;

#ifdef USE_CUDA
/* ------- CUDA version ------- */
	int minGridSize, blockSize, gridSize; /* for optimal call */

	/* allocate array */
	timer = getUnixTime(); /* start to measure time */
	gpuErrchk( cudaMalloc(&x_arr, sizeof(double)*Tlocal) );
	
	if(MPIrank == 0){ /* only master prints */
		std::cout << " - allocation: " << getUnixTime() - timer << "s" << std::endl;
	}
		
	/* fill array */
	if(CALL_NAIVE){
		/* the easiest call */
		timer = getUnixTime();

		for(int k=0;k<nmb_of_tests;k++){
			mykernel<<<1, Tlocal>>>(x_arr, Tlocal, Tstart);
			gpuErrchk( cudaDeviceSynchronize() ); /* synchronize threads after computation */
			MPI_Barrier( MPI_COMM_WORLD ); /* synchronize MPI processes after computation */
		}

		timer1 = getUnixTime() - timer;
	}

	if(CALL_OPTIMAL){
		/* compute optimal parameters of the call */
		gpuErrchk( cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize,mykernel, 0, 0) );
		gridSize = (Tlocal + blockSize - 1)/ blockSize;

		timer = getUnixTime();
			
		for(int k=0;k<nmb_of_tests;k++){
			mykernel<<<blockSize, gridSize>>>(x_arr, Tlocal, Tstart);
			gpuErrchk( cudaDeviceSynchronize() );
			MPI_Barrier( MPI_COMM_WORLD ); 
		}

		timer2 = getUnixTime() - timer;
		for(int k=0;k<MPIsize;k++){
			if(k==MPIrank){
				/* my turn - I am printing */
				std::cout << MPIrank << ". ( gridSize = " << gridSize << ", blockSize = " << blockSize << " )" << std::endl;
			}
			MPI_Barrier( MPI_COMM_WORLD );
		}
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
	timer3 = getUnixTime() - timer;
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
		if(CALL_NAIVE){
			std::cout << " GPU naive   = " << timer1 << "s" << std::endl;
		}

		if(CALL_OPTIMAL){
			std::cout << " GPU optimal = " << timer2 << "s" << std::endl;
		}

#else
		std::cout << " CPU seq    = " << timer3 << "s" << std::endl;
#endif
		std::cout << std::endl;
	}
	MPI_Barrier( MPI_COMM_WORLD );
	
	/* MPI stuff */
	MPI_Finalize();	/* Finalize the MPI environment. */

	return 0;
}


