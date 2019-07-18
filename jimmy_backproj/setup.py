from distutils.core import setup, Extension
import numpy as np

ext_modules = [ Extension('speed', sources = ['speedmodule.c'])]
setup(name='speed_func',
        version='1.0',  \
        include_dirs = [np.get_include()], \
        ext_modules=ext_modules)
