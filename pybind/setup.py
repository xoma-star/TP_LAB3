import os, sys

from distutils.core import setup, Extension
from distutils import sysconfig

cpp_args = ['-std=c++11', '-stdlib=libc++', '-mmacosx-version-min=10.7']

ext_modules = [
    Extension(
    'gzWrapper',
        ['funcs.cpp'],
        include_dirs=['pybind11/include'],
    language='c++',
    extra_compile_args = cpp_args,
    ),
]

setup(
    name='gzWrapper',
    version='1.0.0',
    author='Vladimir Puteen',
    author_email='',
    description='Example',
    ext_modules=ext_modules,
)


# python setup.py build_ext -i