#ifndef _util_cuh
#define _util_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

#if defined(USE_ROCM)
#define cudaUnspecified hipErrorUnknown
#else
#define cudaUnspecified cudaErrorApiFailureBase
#endif

// React to failure on return code != cudaSuccess

#define _cuda_check(fn) \
do { \
    {_cuda_err = fn;} \
    if (_cuda_err != cudaSuccess) goto _cuda_fail; \
} while(false)

// React to failure on return code == 0

#define _alloc_check(fn) \
do { \
    if (!(fn)) { _cuda_err = cudaUnspecified; goto _cuda_fail; } \
    else _cuda_err = cudaSuccess; \
} while(false)

// Clone CPU <-> CUDA

template <typename T>
T* cuda_clone(const void* ptr, int num)
{
    T* cuda_ptr;
    cudaError_t r;

    r = cudaMalloc(&cuda_ptr, num * sizeof(T));
    if (r != cudaSuccess) return NULL;
    r = cudaMemcpy(cuda_ptr, ptr, num * sizeof(T), cudaMemcpyHostToDevice);
    if (r != cudaSuccess) return NULL;
    cudaDeviceSynchronize();
    return cuda_ptr;
}

template <typename T>
T* cpu_clone(const void* ptr, int num)
{
    T* cpu_ptr;
    cudaError_t r;

    cpu_ptr = (T*) malloc(num * sizeof(T));
    if (cpu_ptr == NULL) return NULL;
    r = cudaMemcpy(cpu_ptr, ptr, num * sizeof(T), cudaMemcpyDeviceToHost);
    if (r != cudaSuccess) return NULL;
    cudaDeviceSynchronize();
    return cpu_ptr;
}

// Pack two half values into a half2, host version

__host__ inline __half2 pack_half2(__half h1, __half h2)
{
    unsigned short s1 = *reinterpret_cast<unsigned short*>(&h1);
    unsigned short s2 = *reinterpret_cast<unsigned short*>(&h2);
    ushort2 us2 = make_ushort2(s1, s2);
    return *reinterpret_cast<__half2*>(&us2);
}

#endif