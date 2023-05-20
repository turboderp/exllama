/opt/cuda/bin/nvcc -isystem /opt/cuda/include -isystem /usr/include/python3.10 -D_GLIBCXX_USE_CXX11_ABI=0 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -gencode=arch=compute_89,code=compute_89 -gencode=arch=compute_89,code=sm_89 --compiler-options '-fPIC' -std=c++17 \
test.cu \
../exllama_ext/q4v2_mlp.cu \
../exllama_ext/rms_norm.cu \
../exllama_ext/q4v2_matmul.cu \
../exllama_ext/column_remap.cu \
-o ./test
