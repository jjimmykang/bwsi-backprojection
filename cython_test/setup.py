from distutils.core import setup, Extension
import numpy as np
from Cython.Build import cythonize

setup(name='Hello world app',
      ext_modules=cythonize("hello.pyx"))
