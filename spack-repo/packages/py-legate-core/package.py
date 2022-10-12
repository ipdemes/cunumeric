# Copyright 2013-2022 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

# ----------------------------------------------------------------------------
# If you submit this package back to Spack as a pull request,
# please first remove this boilerplate and all FIXME comments.
#
# This is a template package file for Spack.  We've put "FIXME"
# next to all the things you'll want to change. Once you've handled
# them, you can save this file and test your package like this:
#
#     spack install legate-core-git
#
# You can edit this file again by typing:
#
#     spack edit legate-core-git
#
# See the Spack documentation for more information on packaging.
# ----------------------------------------------------------------------------

from spack.package import *


class PyLegateCore(PythonPackage):
    """FIXME: Put a proper description of your package here."""

    homepage = "https://nv-legate.github.io/legate.core/index.html"
    git = "https://github.com/nv-legate/legate.core.git"

    maintainers = ["ipdemes"]

    version('22.10', branch = "branch-22.10")


    #--------------------------------------------------------------------------#
    # Variants
    #--------------------------------------------------------------------------#

    variant('tests', default=False,
            description='Enable Unit Tests ')

    variant('docs', default=False,
            description='Enable Documentation ')

    variant('prof', default=False,
            description='Enable Prof output')

    variant('shared', default=False,
            description='Build Shared Libraries')

    variant('openmp', default=True,
            description='Build with OpenMP support')

    variant('cuda', default=True,
            description='Build with CUDA support')

    variant ('cuda_arch', default=80, description = 'Cuda architecture')

    variant('shared', default=True,
            description='Build Shared Libraries')


    #--------------------------------------------------------------------------#
    # Dependencies
    #--------------------------------------------------------------------------#

    depends_on('cmake@3.24:')
    depends_on('python@3.8:')
    depends_on('py-pip')
    depends_on('py-scikit-build',type='build')
    depends_on('ninja')
    depends_on('openmpi')
    depends_on('cutensor@1.3.3:')

    cuda_arch_list = ('60', '70', '75', '80', '86')
    for _flag in cuda_arch_list:
        depends_on("nccl cuda_arch=" + _flag, when=" cuda_arch=" + _flag)

    depends_on('py-setuptools@59:', type='build')
    depends_on('py-cffi')
    depends_on('re2')
    depends_on('pcre2')
    depends_on('py-numpy@1.22:')
    depends_on('py-opt-einsum')
    #depends_on('py-pyarrow@5:')
    depends_on('py-scipy')
    depends_on('py-typing-extensions')

#    depends_on('py-colorama', when='+tests')
#    depends_on('py-coverage', when='+tests')
#    depends_on('py-mock', when='+tests')
#    depends_on('py-mypy@0.961:', when='+tests')
#    depends_on('py-pre-commit', when='+tests')
#    depends_on('py-pynvml', when='+tests')
#    depends_on('py-pytest', when='+tests')
#    depends_on('py-pytest-cov', when='+tests')
#    #depends_on('py-pytest-lazy-fixture', when='+tests')
#    depends_on('py-docutils', when='+tests')

    depends_on('legion@cr network=gasnet conduit=mpi  +python +cuda +openmp +redop_complex +bindings +shared ')
    # FIXME: Add dependencies if required.
    # depends_on("foo")

    def install_options(self, spec, prefix):
        options = []
        if '+cuda' in spec:
            options.append('--cuda')
        if 'openmp' in spec:
            options.append('--openmp')
        return options


#    def install(self, spec, prefix):
#        # FIXME: Unknown build system
#        make()
#        make("install")
