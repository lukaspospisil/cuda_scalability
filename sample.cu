#include <iostream>
#include <math.h>
#include <time.h>

/* the length of testing vector */
#define T 100000000

/* the number of levels (number fo subproblems) */
#define LEVELS 20

/* if the lenght of vector is large, set this to zero */
#define PRINT_VECTOR_CONTENT 0

/* which CUDA calls to test? */
#define CALL_NAIVE 1
#define CALL_OPTIMAL 1
#define CALL_TEST 1

/* for measuring time */
double getUnixTime(void){
	struct timespec tv;
	if(clock_gettime(CLOCK_REALTIME, &tv) != 0) return 0;
	return (((double) tv.tv_sec) + (double) (tv.tv_nsec / 1000000000.0));
}

#ifdef USE_CUDA
/* CUDA stuff: */
#include <stdio.h>
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
__global__ void mykernel(double *x_arr, int mysize){
	int i = blockIdx.x*blockDim.x + threadIdx.x; /* compute my id */

	if(i<mysize){ /* maybe we call more than n_local kernels */
		x_arr[i] = i;
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
#ifdef USE_CUDA
	gpuErrchk( cudaDeviceReset() );
#endif

	/* print problem info */
	std::cout << "Benchmark started" << std::endl;
	std::cout << " T          = " << T << std::endl;
	std::cout << " LEVELS     = " << LEVELS << std::endl;
	std::cout << std::endl;

	double timer;
	double times1[LEVELS];
	double times2[LEVELS];
	double times3[LEVELS];
	double times4[LEVELS];
	
	double *x_arr; /* my array on GPU */
	int mysize; /* the lenght of alocated vector (array) */

	for(int level=0; level < LEVELS; level++){
		/* compute the size of testing array on this level */
		mysize = ceil(T/(double)(level+1));
		std::cout << "(" << level+1 << ".): problem of size = " << mysize << std::endl;

#ifdef USE_CUDA
/* ------- CUDA version ------- */

		/* allocate array */
		timer = getUnixTime(); /* start to measure time */
		gpuErrchk( cudaMalloc(&x_arr, sizeof(double)*mysize) );
		std::cout << " - allocation: " << getUnixTime() - timer << "s" << std::endl;
		
		
		/* fill array */
		if(CALL_NAIVE){
			/* the easiest call */
			timer = getUnixTime();

			mykernel<<<1, mysize>>>(x_arr,mysize); 
			gpuErrchk( cudaDeviceSynchronize() ); /* synchronize threads after computation */

			times1[level] = getUnixTime() - timer;
			std::cout << " - call naive: " << times1[level] << " ms" << std::endl;
		}

		if(CALL_OPTIMAL){
			int minGridSize, blockSize, gridSize;
			gpuErrchk( cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize,mykernel, 0, 0) );
			gridSize = (mysize + blockSize - 1)/ blockSize;

			timer = getUnixTime();
			
			mykernel<<<blockSize, gridSize>>>(x_arr, mysize);
			gpuErrchk( cudaDeviceSynchronize() ); 

			times2[level] = getUnixTime() - timer;
			std::cout << " - call optimal: " << times2[level] << " ms" << std::endl;
			std::cout << "   ( gridSize = " << gridSize << ", blockSize = " << blockSize << " )" << std::endl;

		}

		if(CALL_TEST){
			timer = getUnixTime();

			mykernel<<<mysize,1>>>(x_arr,mysize); 
			gpuErrchk( cudaDeviceSynchronize() ); /* synchronize threads after computation */

			times4[level] = getUnixTime() - timer;
			std::cout << " - call test: " << ms << " ms" << std::endl;

		}


		/* print array */
		if(PRINT_VECTOR_CONTENT){
			timer = getUnixTime();

			printkernel<<<1,1>>>(x_arr,mysize);
			gpuErrchk( cudaDeviceSynchronize() );

			std::cout << " - printed in: " << getUnixTime() - timer << "s" << std::endl;
		}

		/* destroy array */
		timer = getUnixTime();
		gpuErrchk( cudaFree(x_arr) );
		std::cout << " - destruction: " << getUnixTime() - timer << "s" << std::endl;

#else
/* ------- SEQUENTIAL version ------- */

		/* allocate array */
		timer = getUnixTime();
		x_arr = new double[mysize];
		std::cout << " - allocation: " << getUnixTime() - timer << "s" << std::endl;

		/* fill array */
		timer = getUnixTime();
		for(int i=0;i<mysize;i++){
			x_arr[i] = i;
		}

		times3[level] = getUnixTime() - timer;
		std::cout << " - call sequential: " << times3[level] << "s" << std::endl;
		
		/* print array */
		if(PRINT_VECTOR_CONTENT){
			timer = getUnixTime();

			std::cout << "  [";
			for(int i=0;i<mysize;i++){
				std::cout << " " << x_arr[i];
				if(i < mysize-1) std::cout << ",";
			}
			std::cout << " ]" << std::endl;

			std::cout << " - printed in: " << getUnixTime() - timer << "s" << std::endl;
		}
		
		/* destroy array */
		timer = getUnixTime();
		delete [] x_arr;
		std::cout << " - destruction: " << getUnixTime() - timer << "s" << std::endl;
#endif

	}


	/* final print of timers */
	std::cout << std::endl;
	std::cout << "---- TIMERS ----" << std::endl;
#ifdef USE_CUDA
	if(CALL_NAIVE){
		std::cout << " GPU naive   = [";
		for(int i=0;i<LEVELS;i++){
			std::cout << " " << times1[i];
			if(i < LEVELS-1) std::cout << ",";
		}
		std::cout << " ]" << std::endl;
	}

	if(CALL_OPTIMAL){
		std::cout << " GPU optimal = [";
		for(int i=0;i<LEVELS;i++){
			std::cout << " " << times2[i];
			if(i < LEVELS-1) std::cout << ",";
		}
		std::cout << " ]" << std::endl;
	}

	if(CALL_TEST){
		std::cout << " GPU test    = [";
		for(int i=0;i<LEVELS;i++){
			std::cout << " " << times4[i];
			if(i < LEVELS-1) std::cout << ",";
		}
		std::cout << " ]" << std::endl;
	}

#else
	std::cout << " CPU seq    = [";
	for(int i=0;i<LEVELS;i++){
		std::cout << " " << times3[i];
		if(i < LEVELS-1) std::cout << ",";
	}
	std::cout << " ]" << std::endl;
#endif
	std::cout << std::endl;

	return 0;
}


