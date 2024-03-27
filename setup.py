from setuptools import setup, Extension
from setuptools import find_packages


VERSION = '0.1.0'

# Install Boost for Ubuntu: sudo apt-get install libboost-all-dev
# the location of the header files is /usr/include/boost

import site, os
site_packages_root_path = site.getsitepackages()[0]

# https://setuptools.pypa.io/en/latest/userguide/ext_modules.html
functions_module = Extension(
    name='cpp_dearth_dataloader',
    sources=['cpp_py_dataloader.cpp'],
    extra_compile_args=["-O3","-fPIC", "-std=c++17"],#, "-lrt"
    include_dirs=[os.path.join(site_packages_root_path, 'pybind11', 'include')],
)


setup(
    name='dearth_dl',  # package name
    version=VERSION,  # package version
    description='For dearth LLM training',  # package description
    packages=find_packages(),  # include all packages under src
    ext_modules=[functions_module],
    install_requires=["numpy", "torch", "pybind11"]
)