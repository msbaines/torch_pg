import logging
import os
import torch

from packaging import version
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension


root_dir = os.path.dirname(os.path.abspath(__file__))


mpi_home = os.environ.get("MPI_HOME")
if mpi_home is None:
    mpi_home = "/usr/lib/x86_64-linux-gnu/openmpi/"
if not os.path.exists(mpi_home):
    print("Couldn't find MPI install dir, please set MPI_HOME env variable")
    sys.exit(1)


nccl_home = os.environ.get("NCCL_HOME")
if nccl_home is None or not os.path.exists(nccl_home):
    nccl_home = None
    logging.warn("Couldn't find NCCL install dir, please set NCCL_HOME to enable NCCL build")


torch_version = version.parse(torch.__version__)
torch_version_defines = ["-DTORCH_MAJOR="+str(torch_version.major), "-DTORCH_MINOR="+str(torch_version.minor)]


extensions = []
cmdclass = {}

extensions = [
    CppExtension(
        name="torch_pg._C",
        sources=[
            "src/Bindings.cpp",
            "src/ProcessGroupMPI.cpp",
        ],
        include_dirs=[
            os.path.join(root_dir, "include"), 
            os.path.join(mpi_home, "include"),
        ],
        library_dirs=[
            os.path.join(mpi_home, "lib"),
        ],
        libraries=["mpi",],
        extra_compile_args=["-DOMPI_SKIP_MPICXX=1"] + torch_version_defines,
    ),
]

if nccl_home is not None:
    extensions += [
        CUDAExtension(
            name="torch_pg._CUDA",
            sources=[
                "src/CUDABindings.cpp",
                "src/NCCLUtils.cpp",
                "src/ProcessGroupNCCL.cpp"
            ],
            include_dirs=[
                os.path.join(root_dir, "include"),
                os.path.join(nccl_home, "include"),
            ],
            library_dirs=[
                os.path.join(nccl_home, "lib"),
            ],
            libraries=["nccl",],
            extra_compile_args=["-DENABLE_NCCL_P2P_SUPPORT"] + torch_version_defines,
        ),
    ]

cmdclass["build_ext"] = BuildExtension


setup(
    name="torch-pg",
    version="0.0.0",
    packages=find_packages(),
    ext_modules=extensions,
    cmdclass=cmdclass,
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: BSD License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
    ],
)
