#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <chrono>
#include <iostream>
#include <sstream>
#include <locale>
#include <iomanip>
#include <limits>

#include "../exllama_ext/util.h"
#include "../exllama_ext/matrix.h"
#include "../exllama_ext/q4v2_mlp.h"

using namespace std;

template <typename T>
class Tensor
{
public:
    T* data_cuda;
    T* data_cpu;
    uint32_t height;
    uint32_t width;

    // From file

    Tensor(const char* filename)
    {
        FILE* file = fopen(filename, "rb");
        if (!file)
        {
            cout << "File not found: " << filename << "\n";
            return;
        }

        fseek(file, 0, SEEK_END);
        long size = ftell(file);
        fseek(file, 0, SEEK_SET);
        size_t num_elements = size / sizeof(T);

        if (num_elements == 0)
        {
            data_cuda = NULL;
            data_cpu = NULL;
            height = 0;
            width = 0;

            cout << " ** " << filename << " (None)\n";
            return;
        }

        data_cpu = new T[num_elements];
        fread(data_cpu, sizeof(T), num_elements, file);
        fclose(file);

        char filenameshape[1024];
        strcpy(filenameshape, filename);
        strcat(filenameshape, ".shape");

        file = fopen(filenameshape, "rb");
        if (!file)
        {
            cout << "File not found: " << filenameshape << "\n";
            return;
        }

        fread(&height, 1, sizeof(uint32_t), file);
        fread(&width, 1, sizeof(uint32_t), file);
        fclose(file);

        if (width * height != num_elements)
        {
            cout << "Incorrect shape: " << filenameshape << "\n";
            return;
        }

        cudaMalloc(&data_cuda, size);
        dataToCUDA();

        cout << " ** " << filename << " (" << height << ", " << width << ")\n";
    }

    // Empty tensor

    Tensor(int _height, int _width)
    {
        height = _height;
        width = _width;

        size_t size = (height * width) * sizeof(T);
        cudaMalloc(&data_cuda, size);

        data_cpu = new T[height * width];
    }

    // Zero tensor

    Tensor(int _height, int _width, T zero_value)
    {
        height = _height;
        width = _width;

        size_t size = (height * width) * sizeof(T);
        cudaMalloc(&data_cuda, size);

        data_cpu = new T[height * width];

        for (int i = 0; i < _width * _height; i++) data_cpu[i] = zero_value;
        dataToCUDA();
    }

    // Fill

    void fill(T value)
    {
        for (int i = 0; i < width * height; i++) data_cpu[i] = value;
        dataToCUDA();
    }

    // Copy data

    void dataToCUDA()
    {
        size_t size = (height * width) * sizeof(T);
        cudaMemcpy(data_cuda, data_cpu, size, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
    }

    void dataToCPU()
    {
        size_t size = (height * width) * sizeof(T);
        cudaMemcpy(data_cpu, data_cuda, size, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
    }

};

__global__ void dummyKernel()
{
    // Dummy kernel
}

void warmUpCUDA()
{
    // Create a CUDA context
    cudaFree(0);

    // Launch a dummy kernel
    dummyKernel<<<1, 1>>>();
    cudaDeviceSynchronize();
}


template <typename T>
float compareTensors(Tensor<T>& a, Tensor<T>& b, int height = 0, int width = 0)
{
    if (height == 0 && (a.width != b.width || a.height != b.height))
    {
        cout << "Incompatible sizes.\n";
        return std::numeric_limits<float>::infinity();
    }

    if (height == 0) height = a.height;
    if (width == 0) width = a.width;

    a.dataToCPU();
    b.dataToCPU();

    float m = 0.0f;

    for (int r = 0; r < height; r++)
    {
        for (int c = 0; c < width; c++)
        {
            float a_f = __half2float(a.data_cpu[r * a.width + c]);
            float b_f = __half2float(b.data_cpu[r * b.width + c]);
            m = fmax(m, fabs(a_f - b_f));
        }
    }

    return m;
}

void printTensor(Tensor<half>& a)
{
    int width = 8; if (width > a.width) width = a.width;
    int height = 8; if (height > a.height) height = a.height;

    a.dataToCPU();

    for (int c = 0; c < width; c++) cout << "---------";
    cout << "\n";

    for (int r = 0; r < height; r++)
    {
        for (int c = 0; c < width; c++)
        {
            float a_f = __half2float(a.data_cpu[r * a.width + c]);
            cout << setfill(' ') << setprecision(5) << setw(9) << a_f << dec;
        }
        cout << "\n";
    }
}

void printTensor(Tensor<uint32_t>& a)
{
    int width = 8; if (width > a.width) width = a.width;
    int height = 8; if (height > a.height) height = a.height;

    a.dataToCPU();

    for (int c = 0; c < width; c++) cout << "---------";
    cout << "\n";

    for (int r = 0; r < height; r++)
    {
        for (int c = 0; c < width; c++)
        {
            uint32_t a_i = a.data_cpu[r * a.width + c];
            cout << " " << setfill('0') << setw(8) << hex << a_i << dec << setfill(' ');
        }
        cout << "\n";
    }
}


int main()
{
    warmUpCUDA();
    int iters;

    cout << fixed << setprecision(6);
    cout << "Loading tensors...\n";

    // Test MLP

    Tensor<half>        x                    ("mlp/test_mlp_x");
    Tensor<half>        x_gated              ("mlp/test_mlp_x_gated");
    Tensor<half>        x_done               ("mlp/test_mlp_x_done");
    Tensor<half>        x_prenorm            ("mlp/test_mlp_x_prenorm");
    Tensor<half>        x_postresidual       ("mlp/test_mlp_x_postresidual");

    Tensor<half>        rms_norm_weight      ("mlp/test_mlp_norm_weight");

    Tensor<half>        up_proj_bias         ("mlp/up_proj.bias");
    Tensor<uint32_t>    up_proj_qweight      ("mlp/up_proj.qweight");
    Tensor<uint32_t>    up_proj_qzeros       ("mlp/up_proj.qzeros");
    Tensor<half>        up_proj_scales       ("mlp/up_proj.scales");
    Tensor<uint16_t>    up_proj_seq_g_idx    ("mlp/up_proj.seq_g_idx");
    Tensor<uint32_t>    up_proj_x_map        ("mlp/up_proj.x_map");

    Tensor<half>        gate_proj_bias       ("mlp/gate_proj.bias");
    Tensor<uint32_t>    gate_proj_qweight    ("mlp/gate_proj.qweight");
    Tensor<uint32_t>    gate_proj_qzeros     ("mlp/gate_proj.qzeros");
    Tensor<half>        gate_proj_scales     ("mlp/gate_proj.scales");
    Tensor<uint16_t>    gate_proj_seq_g_idx  ("mlp/gate_proj.seq_g_idx");
    Tensor<uint32_t>    gate_proj_x_map      ("mlp/gate_proj.x_map");

    Tensor<half>        down_proj_bias       ("mlp/down_proj.bias");
    Tensor<uint32_t>    down_proj_qweight    ("mlp/down_proj.qweight");
    Tensor<uint32_t>    down_proj_qzeros     ("mlp/down_proj.qzeros");
    Tensor<half>        down_proj_scales     ("mlp/down_proj.scales");
    Tensor<uint16_t>    down_proj_seq_g_idx  ("mlp/down_proj.seq_g_idx");
    Tensor<uint32_t>    down_proj_x_map      ("mlp/down_proj.x_map");

    Tensor<half> x_temp(x.height, x.width);
    Tensor<float> x_col_temp(1, x.height);
    Tensor<half> x_act_temp(x.height, gate_proj_qweight.width);

    Tensor<half> out(x_gated.height, x_gated.width);

    int groupsize = gate_proj_qweight.height * 8 / gate_proj_qzeros.height;

    iters = 1;
    auto start_time = chrono::high_resolution_clock::now();

    cout << "--------\n";

    cout << "Fused MLP (" << iters << " iterations)... ";

    for (int i = 0; i < iters; i++)
    {
        q4v2_mlp_cuda
        (
            x_prenorm.data_cuda,            // input

            x_temp.data_cuda,               // input, normalized (empty)
            x_col_temp.data_cuda,           // temp for norm (empty)
            x_act_temp.data_cuda,           // temp for act(x @ gate) * x @ up (empty)

            rms_norm_weight.data_cuda,
            (1e-06),

            gate_proj_qweight.data_cuda,
            gate_proj_scales.data_cuda,
            gate_proj_qzeros.data_cuda,
            gate_proj_seq_g_idx.data_cuda,
            gate_proj_x_map.data_cuda,

            up_proj_qweight.data_cuda,
            up_proj_scales.data_cuda,
            up_proj_qzeros.data_cuda,
            up_proj_seq_g_idx.data_cuda,
            up_proj_x_map.data_cuda,

            down_proj_qweight.data_cuda,
            down_proj_scales.data_cuda,
            down_proj_qzeros.data_cuda,
            down_proj_seq_g_idx.data_cuda,
            down_proj_x_map.data_cuda,

            x.height,
            x.width,
            gate_proj_qweight.width,
            groupsize
        );
    }

    cudaDeviceSynchronize();

    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();
    duration /= iters;
    cout << duration << " us / iteration\n";

    cout << "Validating fused MLP... ";

    float diff = compareTensors<half>(x_prenorm, x_postresidual);

    cout << "max diff.: " << diff <<"\n";

    printTensor(x_prenorm);
    printTensor(x_postresidual);

    printf("Done\n");
    return 0;
}
