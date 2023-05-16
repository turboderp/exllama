#include "q4v2_sequential.h"

const int UNSHUF_BLOCKSIZE_X = 64;

template <typename T>
T* cuda_clone(const void* ptr, int num)
{
    T* cuda_ptr;
    size_t size = num * sizeof(T);
    cudaMalloc(&cuda_ptr, size);
    cudaMemcpy(cuda_ptr, ptr, size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    return cuda_ptr;
}

template <typename T>
T* cpu_clone(const void* ptr, int num)
{
    size_t size = num * sizeof(T);
    T* cpu_ptr = (T*) malloc(size);
    cudaMemcpy(cpu_ptr, ptr, size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    return cpu_ptr;
}

__global__ void q4v2_sequential_kernel
(
    const uint32_t* w,
    uint32_t* w_new,
    const uint32_t* x_map,
    const int w_height,
    const int w_width
)
{
    const uint64_t* w2 = (uint64_t*) w;
    uint64_t* w_new2 = (uint64_t*) w_new;
    int w2_stride = w_width >> 1;

    int w2_column = UNSHUF_BLOCKSIZE_X * blockIdx.x + threadIdx.x;
    int w_new2_row = blockIdx.y;

    int x_map_idx = w_new2_row << 3;

    uint64_t dst = 0;

    #pragma unroll
    for (int i = 0; i < 8; i++)
    {
        int source_row = x_map[x_map_idx++];

        int w2_row = source_row >> 3;
        int w2_subrow = source_row & 0x07;
        int w2_row_shift = w2_subrow << 2;
        int wnew2_row_shift = i << 2;

        uint64_t src = w2[w2_row * w2_stride + w2_column];
        src >>= w2_row_shift;
        src &= 0x0000000f0000000f;
        src <<= wnew2_row_shift;
        dst |= src;
    }

    w_new2[w_new2_row * w2_stride + w2_column] = dst;
}


// Unshuffle rows in w to sequentialize group index
//
// w is overwritten with seq_w
// seq_g_idx will contain ushort pairs of { group_number, remaining_items_in_group_inclusive, ... }
// x_map will be the column remapping to perform x -> seq_x such that seq_x @ seq_w == x @ w

void q4v2_sequential_cuda
(
    uint32_t* w,
    const int w_height,
    const int w_width,
    const uint32_t* g_idx,  // size: w_height * 8
    uint16_t* seq_g_idx,    // size: w_height * 8 * 2
    uint32_t* x_map_cuda,   // size: w_height * 8
    const int num_groups
)
{
    // Buffers

    uint32_t* w_new;
    int res = cudaMalloc(&w_new, w_height * w_width * sizeof(uint32_t));

    uint32_t* g_idx_cpu = cpu_clone<uint32_t>(g_idx, w_height * 8);
    uint16_t* seq_g_idx_cpu = (uint16_t*) calloc(w_height * 8 * 2, sizeof(uint16_t));
    uint32_t* x_map = (uint32_t*) malloc(w_height * 8 * sizeof(uint32_t));

    // Get no. groups

//     int max_g_idx = 0;
//     for (int i = 0; i < w_height; ++i) if (g_idx_cpu[i] > max_g_idx) max_g_idx = g_idx_cpu[i];
//     max_g_idx++;
//     printf("\n-> %i\n", max_g_idx);

    int max_g_idx = num_groups;
    uint32_t* g_idx_map = (uint32_t*) calloc(max_g_idx, sizeof(uint32_t));

    // Group histogram

    for (int i = 0; i < w_height * 8; i++) g_idx_map[g_idx_cpu[i]]++;

    // Create new group index

    int r = 0;

    for (int g = 0; g < max_g_idx; g++)
    {
        int r_rem = g_idx_map[g];
        int r_max = r + r_rem;
        for (; r < r_max; r++)
        {
            seq_g_idx_cpu[r * 2] = g;
            seq_g_idx_cpu[r * 2 + 1] = r_rem--;
        }
    }

    // Group map

    for (int i = 0, acc = 0; i < max_g_idx; i++)
    {
        short tmp = g_idx_map[i];
        g_idx_map[i] = acc;
        acc += tmp;
    }

    // X map (inverse)

    uint32_t* x_map_inv = (uint32_t*) malloc(w_height * 8 * sizeof(uint32_t));

    for (int row = 0; row < w_height * 8; row++)
    {
        uint32_t target_group = g_idx_cpu[row];
        uint32_t target_row = g_idx_map[target_group];
        g_idx_map[target_group]++;
        x_map_inv[row] = target_row;
    }

    // X map

    for (int row = 0; row < w_height * 8; row++)
    {
        x_map[x_map_inv[row]] = row;
    }

    // Move to CUDA

    cudaMemcpy(x_map_cuda, x_map, w_height * 8 * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // Rearrange rows in w

    dim3 threads
    (
        UNSHUF_BLOCKSIZE_X,
        1,
        1
    );

    dim3 blocks
    (
        w_width / UNSHUF_BLOCKSIZE_X / 2,
        w_height,
        1
    );

    cudaDeviceSynchronize();

    q4v2_sequential_kernel<<<blocks, threads>>>
    (
        w,
        w_new,
        x_map_cuda,
        w_height,
        w_width
    );

    cudaDeviceSynchronize();

    cudaMemcpy(w, w_new, w_height * w_width * sizeof(uint32_t), cudaMemcpyDeviceToDevice);
    cudaMemcpy(seq_g_idx, seq_g_idx_cpu, w_height * 8 * 2 * sizeof(uint16_t), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    cudaFree(w_new);

    free(g_idx_cpu);
    free(seq_g_idx_cpu);

    free(g_idx_map);
    free(x_map);
    free(x_map_inv);

    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(err));

    // for (int row = 0; row < w_height * 8; row++)
    // {
    //     uint32_t w_row = row >> 3;
    //     uint32_t w_subrow = row & 0x07;
    //     uint32_t w_row_shift = w_subrow << 2;
    //     uint32_t w_idx = w_row * w_width;
    //
    //     uint32_t target_row = x_map_inv[row];
    //
    //     uint32_t target_w_row = target_row >> 3;
    //     uint32_t target_w_subrow = target_row & 0x07;
    //     uint32_t target_w_row_shift = target_w_subrow << 2;
    //     uint32_t target_w_idx = target_w_row * w_width;
    //
    //     #pragma unroll
    //     for (int column = 0; column < w_width; column++)
    //     {
    //         uint32_t v = w[w_idx++];
    //         v >>= w_row_shift;
    //         v &= 0x0f;
    //         v <<= target_w_row_shift;
    //         w_new[target_w_idx++] |= v;
    //     }
    // }
}




