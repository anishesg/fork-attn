from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

src_dir = os.path.join(os.path.dirname(__file__), "src")

setup(
    name="fork_attn",
    version="0.1.0",
    description="Fused tree-structured decode attention with shared-prefix KV computation",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="fork_attn._C",
            sources=[
                "src/torch_binding.cpp",
                "src/shared_prefix_attn.cu",
                "src/branch_suffix_attn.cu",
                "src/fork_attn.cu",
                "src/naive_decode_attn.cu",
            ],
            include_dirs=[src_dir],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-std=c++17",
                    "-gencode=arch=compute_80,code=sm_80",
                    "-gencode=arch=compute_86,code=sm_86",
                    "-gencode=arch=compute_89,code=sm_89",
                    "-gencode=arch=compute_90,code=sm_90",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.8",
    install_requires=["torch"],
)
