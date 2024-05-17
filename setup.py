from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(name="cuda_solver",
              sources=["neo_solver.pyx"],
              libraries=[":neo_solver.so"],
              library_dirs=["."],
              language='c++',
              include_dirs=[numpy.get_include()],
              extra_link_args=['-Wl,-rpath=.'])
]

setup(
    name='cuda_solver',
    ext_modules=cythonize(ext_modules),
)