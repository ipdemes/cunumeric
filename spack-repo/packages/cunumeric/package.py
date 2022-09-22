from spack import *


class Cunumeric(CMakePackage, CudaPackage):
    '''cuNumeric is a Legate library that aims to provide a distributed
    and accelerated drop-in replacement for the NumPy API on top of the
    Legion runtime.
    '''
    homepage = 'https://nv-legate.github.io/cunumeric/index.html'
    git      = 'https://github.com/nv-legate/cunumeric'

    version('22.10', branch='branch-22.10', submodules=False, preferred=False)


    #--------------------------------------------------------------------------#
    # Variants
    #--------------------------------------------------------------------------#

    variant('tests', default=False,
            description='Enable Unit Tests ')

    variant('docs', default=False,
            description='Enable Documentation ')

    variant('prof', default=False,
            description='Enable Prof output')

    variant('shared', default=True,
            description='Build Shared Libraries')

    #variant ('cuda_arch', default=None, description = 'Cuda architecture')

    #--------------------------------------------------------------------------#
    # Dependencies
    #--------------------------------------------------------------------------#

    # CMake

    depends_on('cmake@3.24:')

    #python

    depends_on('python@3.10')

    #pip

    depends_on('py-pip')

    #git

    depends_on('git')

    #zlib

    depends_on('zlib')

    #ninja

    depends_on('ninja')

    #openmpi

    depends_on('openmpi')

    #scikit-build

    depends_on('py-scikit-build')

    #cutensor

    depends_on('cutensor@1.3.3:')

    #nccl

    cuda_arch_list = ('60', '70', '75', '80', '86', '89', '90')
    for _flag in cuda_arch_list:
        depends_on("nccl cuda_arch=" + _flag, when=" cuda_arch=" + _flag)
    #depends_on('nccl cuda_arch=' )

    #setuptools

    depends_on('py-setuptools@59:')

    #cffi

    depends_on('py-cffi')

    #re2

    depends_on('re2')
    depends_on('pcre2')

    #numpy

    depends_on('py-numpy@1.22:')

    #opt_einsum

    depends_on('py-opt-einsum')

    #pyarrow

    depends_on('py-pyarrow@5:')

    #scipy

    depends_on('py-scipy')

    #typing_extensions

    depends_on('py-typing-extensions')

    #clang

    depends_on('llvm')

    #llvm-openmp

    depends_on('llvm-openmp')

    # tests

    #clang-tools

    #depends_on('clang-tools@8:', when='+tests')

    #colorama
    
    depends_on('py-colorama', when='+tests')
    
    #coverage
    
    depends_on('py-coverage', when='+tests')

    #mock
    
    depends_on('py-mock', when='+tests')

    #mypy

    depends_on('py-mypy@0.961:', when='+tests')

    #pre-commit

    depends_on('py-pre-commit', when='+tests')

    #pynvml

    depends_on('py-pynvml', when='+tests')

    #pytest

    depends_on('py-pytest', when='+tests')

    #pytests-cov

    depends_on('py-pytest-cov', when='+tests')

    #pytest-lazy-fixture

    #depends_on('py-pytest-lazy-fixture', when='+tests')

    #types-docutils

    depends_on('py-docutils', when='+tests')

    #Docs

    #jinja2

    depends_on('py-jinja2', when='+docs')

    #pydata-sphinx-theme

    #depends_on('py-pydata-sphinx-theme', when='+docs')

    #recommonmark
    
    depends_on('py-recommonmark', when='+docs')

    #markdown

    depends_on('py-markdown@:3.4.0', when='+docs')

    #sphinx
    depends_on('py-sphinx@4.4.0:', when='+docs')

    #sphinx-copybutton

    depends_on('py-sphinx-copybutton', when='+docs')

    #sphinx-markdown-tables

    #depends_on('py-sphinx-markdown-tables', when='+docs')

    # Graphviz

    depends_on('graphviz', when='+prof')



