from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
numpy_include=numpy.get_include()
'''
ext_modules=[
    Extension(
        #"cython_bbox",
        ["bbox.pyx"],
        include_dirs=[numpy_include]
        ),
    ]
setup(
    ext_modules = ext_modules,
    #cmdclass={'build_ext':custom_build_ext},

)
'''
setup(ext_modules=cythonize("bbox.pyx"),include_dirs=[numpy_include])
