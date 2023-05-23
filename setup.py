from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(
    name="exllama",
    version="0.0.1",
    install_requires=[
        "torch",
    ],
    packages=["exllama"],
    py_modules=["exllama"],
    ext_modules=[
        cpp_extension.CUDAExtension(
            "exllama_ext",
            [
                "exllama_ext/column_remap.cu",
                "exllama_ext/exllama_ext.cpp",
                "exllama_ext/q4v2_matmul.cu",
                "exllama_ext/q4v2_mlp.cu",
                "exllama_ext/q4v2_recons.cu",
                "exllama_ext/q4v2_sequential.cu",
                "exllama_ext/rms_norm.cu",
            ],
            extra_compile_args={"nvcc": ["-O3"]},
        ),
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
