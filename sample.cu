#include <iostream>
#include <math.h>


/* the length of testing vector */
#define T 10

/* the number of levels (number fo subproblems) */
#define LEVELS 3

/* if the lenght of vector is large, set this to zero */
#define PRINT_VECTOR_CONTENT 1

/* which CUDA calls to test? */
#define CALL_NAIVE 1
#define CALL_OPTIMAL 1

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
	/* if i >= mysize then relax and do nothing */	
}

/* print kernel, call only one (!)  */
/* in practical applications, this way is a real bottleneck, array should be transfered to CPU and printed there */
/* anyway, in this sample I want to print only small arrays, so who cares.. */
__global__ void printkernel(double *x_arr, int mysize){
	printf("  [");
	for(int i=0;i<mysize;i++){
		printf(" %d", x_arr[i]);
		if(i < mysize-1) printf(",");
	}
	printf(" ]\n");
}


/* end of CUDA stuff */
#endif



int main( int argc, char *argv[] )
{
	/* print problem info */
	std::cout << "Benchmark started" << std::endl;
	std::cout << " T          = " << T << std::endl;
	std::cout << " LEVELS     = " << LEVELS << std::endl;
	std::cout << std::endl;
		
	double *x_arr; /* my array on GPU */
	int mysize; /* the lenght of alocated vector (array) */

	for(int level=0; level < LEVELS; level++){
		/* compute the size of testing array on this level */
		mysize = ceil(T/(level+1));
		std::cout << "(" << level+1 << ".): problem of size = " << mysize << std::endl;

#ifdef USE_CUDA
/* ------- CUDA version ------- */

		/* allocate array */
		gpuErrchk( cudaMalloc(&x_arr, sizeof(double)*mysize) );
		
		/* fill array */
		if(CALL_NAIVE){
			/* the easiest call */
			mykernel<<<mysize, 1>>>(x_arr,mysize); 
			gpuErrchk( cudaDeviceSynchronize() ); /* synchronize threads after computation */
		}

		if(CALL_OPTIMAL){
			int minGridSize, blockSize, gridSize;
			gpuErrchk( cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize,mykernel, 0, 0) );
			gridSize = (mysize + blockSize - 1)/ blockSize;
			std::cout << " - gridSize = " << gridSize << ", blockSize = " << blockSize << std::endl;
			
			mykernel<<<gridSize, blockSize>>>(x_arr, mysize);
			gpuErrchk( cudaDeviceSynchronize() ); 
		}

		/* print array */
		if(PRINT_VECTOR_CONTENT){
			printkernel<<<1,1>>>(x_arr,mysize);
			gpuErrchk( cudaDeviceSynchronize() );
		}

		/* destroy array */
		gpuErrchk( cudaFree(&x_arr) );

#else
/* ------- SEQUENTIAL version ------- */

		/* allocate array */
		x_arr = new double[mysize];

		/* fill array */
		for(int i=0;i<mysize;i++){
			x_arr[i] = i;
		}
		
		/* print array */
		if(PRINT_VECTOR_CONTENT){
			std::cout << "  [";
			for(int i=0;i<mysize;i++){
				std::cout << " " << x_arr[i];
				if(i < mysize-1) std::cout << ",";
			}
			std::cout << " ]" << std::endl;
		}
		
		/* destroy array */
		delete [] x_arr;
#endif

	}



	return 0;
}


