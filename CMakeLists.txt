# options:
# -DUSE_CUDA=ON				-> compile with GPU
#

project(CUDA_SCALABILITY)
cmake_minimum_required(VERSION 2.8)

option(USE_CUDA "USE_CUDA" OFF)

# CMAKE: include cmake functions
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/util/" ${CMAKE_MODULE_PATH})

# give some info about Petsc
message(STATUS "CMAKE_CXX_COMPILER = ${CMAKE_CXX_COMPILER}")
message(STATUS "USE_CUDA = ${USE_CUDA}")

# include CUDA stuff
if(${USE_CUDA})
	include(FindCUDA)
	set(CUDA_PROPAGATE_HOST_FLAGS off) # if flags are passed with -Xcompiler, they also affect NVCC which doesn't understand all g++ flags we use
	set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER}) # without this, cc is used instead of CC and all include paths have to be specified manually
	string(TOUPPER "${CMAKE_BUILD_TYPE}" BUILD_TYPE_UPPER)
	set(CUDA_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-vla ${CMAKE_CXX_FLAGS_${BUILD_TYPE_UPPER}}") # add flags specific to build type
	string(REPLACE "-std=c++11" "" CUDA_CXX_FLAGS ${CUDA_CXX_FLAGS}) # remove C++11 from options

	# send USE_CUDA to compiled code
	set(CUDA_CXX_FLAGS "${CUDA_CXX_FLAGS} -DUSE_CUDA")
endif()

# compile the code
if(${USE_CUDA})
	# compile using cuda compiler
	cuda_add_executable(sample sample.cu
			OPTIONS "-arch=sm_35 --compiler-options \"${CUDA_CXX_FLAGS}\""
			DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
else ()
	# compile using standart c++ compiler
	add_executable(sample sample.cpp)
endif()
