from distutils.core import setup, Extension
import numpy as np
from Cython.Build import cythonize

setup(name='Backproject',
      ext_modules=cythonize("backproj_c.pyx"))
