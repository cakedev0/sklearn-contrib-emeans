from Cython.Build import cythonize
from setuptools import Extension, setup

setup(
    ext_modules=cythonize(
        [
            Extension(
                "hkm_kernels",
                ["hkm_kernels.pyx"],
                extra_compile_args=["-O3", "-fopenmp", "-march=native"],
                extra_link_args=["-fopenmp"],
            )
        ]
    )
)
