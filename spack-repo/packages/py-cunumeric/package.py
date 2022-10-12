
from spack.package import *


class PyCunumeric(PythonPackage):
    '''cuNumeric is a Legate library that aims to provide a distributed
    and accelerated drop-in replacement for the NumPy API on top of the
    Legion runtime.
    '''
    homepage = 'https://nv-legate.github.io/cunumeric/index.html'
    git      = 'https://github.com/nv-legate/cunumeric.git'

    version('22.10', branch='branch-22.10')


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

    variant('openmp', default=True,
            description='Build with OpenMP support')

    variant('cuda', default=True,
            description='Build with CUDA support')

    #variant ('cuda_arch', default=None, description = 'Cuda architecture')

    #--------------------------------------------------------------------------#
    # Dependencies
    #--------------------------------------------------------------------------#

    depends_on('cmake@3.24:')
    depends_on('python@3.8:')
    depends_on('py-pip')
    depends_on('py-scikit-build', type='build')
    #depends_on('git')
    #depends_on('zlib')
    depends_on('ninja')
    depends_on('openmpi')
    depends_on('cutensor@1.3.3:')

#    cuda_arch_list = ('60', '70', '75', '80', '86')
#    for _flag in cuda_arch_list:
#        depends_on("nccl cuda_arch=" + _flag, when=" cuda_arch=" + _flag)

    depends_on('py-setuptools@59:',type='build')
    depends_on('py-cffi')
    depends_on('re2')
    depends_on('pcre2')
    depends_on('py-numpy@1.22:')
    depends_on('py-opt-einsum')
    depends_on('utf8proc@2.2:')
    depends_on('py-pyarrow@5:')
    depends_on('py-scipy')
    depends_on('py-typing-extensions')

    #clang

    #depends_on('llvm')

    #llvm-openmp

    #depends_on('llvm-openmp')

    # tests

    #clang-tools

    #depends_on('clang-tools@8:', when='+tests')

    
    depends_on('py-colorama', when='+tests')
    depends_on('py-coverage', when='+tests')
    depends_on('py-mock', when='+tests')
    depends_on('py-mypy@0.961:', when='+tests')
    depends_on('py-pre-commit', when='+tests')
    depends_on('py-pynvml', when='+tests')
    depends_on('py-pytest', when='+tests')
    depends_on('py-pytest-cov', when='+tests')
    #depends_on('py-pytest-lazy-fixture', when='+tests')
    depends_on('py-docutils', when='+tests')

    #Docs

    depends_on('py-jinja2', when='+docs')
    #depends_on('py-pydata-sphinx-theme', when='+docs')
    depends_on('py-recommonmark', when='+docs')
    depends_on('py-markdown@:3.4.0', when='+docs')
    depends_on('py-sphinx@4.4.0:', when='+docs')
    depends_on('py-sphinx-copybutton', when='+docs')
    #depends_on('py-sphinx-markdown-tables', when='+docs')
    depends_on('graphviz', when='+prof')


    #FIXME
    depends_on('py-legate-core +shared cuda_arch=80')
#    depends_on('legion@cr network=gastet conduit=mpi  +python +cuda +openmpi +redop_complexi +bindings arch=80')

 #   def global_options(self, spec, prefix):
 #       options = []
 #       if '+cuda' in spec:
 #           options.append('--cuda')
 #       if 'openmp' in spec:
 #           options.append('--openmp')
 #       return options

