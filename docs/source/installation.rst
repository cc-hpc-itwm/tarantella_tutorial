.. _installation-label:

Installation
============

Tarantella needs to be built `from source <https://github.com/cc-hpc-itwm/tarantella>`_.
Since Tarantella is built on top of `TensorFlow <https://www.tensorflow.org/>`_,
you will require a recent version of it. Additionally, you will need an installation of
the open-source communication libraries `GaspiCxx <https://github.com/cc-hpc-itwm/GaspiCxx>`_
and `GPI-2 <http://www.gpi-site.com/>`_,
which Tarantella uses to implement distributed training.

Lastly, you will need `pybind11 <https://github.com/pybind/pybind11>`_, which is required
for Python and C++ inter-communication.

In the following we will look at the required steps in detail.

Installing dependencies
-----------------------

Compiler and build system
^^^^^^^^^^^^^^^^^^^^^^^^^

Tarantella can be built using a recent `gcc <https://gcc.gnu.org/>`_
compiler with support for C++17 (starting with ``gcc 7.4.0``).
You will also need the build tool `CMake <https://cmake.org/>`_ (from version ``3.12``).

Installing TensorFlow
^^^^^^^^^^^^^^^^^^^^^

First you will need to install TensorFlow.
Supported versions start at ``Tensorflow 2.4``, and they can be installed in a conda
environment using pip, as recommended on the
`TensorFlow website <https://www.tensorflow.org/install>`_.

.. caution::

  This tutorial targets installations on the STYX cluster, where some of the dependencies are pre-installed.
  For a full description of Tarantella's installation steps, refer to the 
  `Tarantella documentation <https://tarantella.readthedocs.io/en/latest/installation.html>`_.

To get started, create and activate an environment for Tarantella:

.. code-block:: bash

  conda create -n tarantella
  conda activate tarantella

Now, you can install the latest supported TensorFlow version with:

.. code-block:: bash

  conda install -c nvidia python==3.9 cudatoolkit~=11.2 cudnn
  pip install --upgrade tensorflow_gpu
  conda install pybind11 pytest networkx

Tarantella requires at least Python ``3.7``. Make sure the selected version also matches
the `TensorFlow requirements <https://www.tensorflow.org/install>`_.

.. _gaspicxx-install-label:

Installing GaspiCxx
^^^^^^^^^^^^^^^^^^^

`GaspiCxx <https://github.com/cc-hpc-itwm/GaspiCxx>`_ is a C++ abstraction layer built
on top of the GPI-2 library, designed to provide easy-to-use point-to-point and collective
communication primitives.
Tarantella's communication layer is based on GaspiCxx and its
`PyGPI <https://github.com/cc-hpc-itwm/GaspiCxx/blob/v1.2.0/src/python/README.md>`_ API for Python.

To install GaspiCxx and PyGPI, first download the latest ``dev`` branch from the
`git repository <https://github.com/cc-hpc-itwm/GaspiCxx>`_:

.. code-block:: bash

  git clone https://github.com/cc-hpc-itwm/GaspiCxx.git
  cd GaspiCxx
  git checkout -b dev

Compile and install the library as follows, making sure the previously created conda
environment is activated:

.. code-block:: bash

  conda activate tarantella

  mkdir build && cd build
  export GASPICXX_INSTALLATION_PATH=/your/gaspicxx/installation/path
  cmake -DBUILD_PYTHON_BINDINGS=ON    \
        -DBUILD_SHARED_LIBS=ON        \
        -DCMAKE_INSTALL_PREFIX=${GASPICXX_INSTALLATION_PATH} ../
  make -j$(nproc) install

where ``${GASPICXX_INSTALLATION_PATH}`` needs to be set to the path where you want to install
the library.

Building Tarantella from source
-------------------------------

With all dependencies installed, we can now download, configure and compile Tarantella.
To download the source code, simply clone the
`GitHub repository <https://github.com/cc-hpc-itwm/tarantella.git>`__:

.. code-block:: bash

  git clone https://github.com/cc-hpc-itwm/tarantella.git
  cd tarantella
  git checkout master

Next, we need to configure the build system using CMake.
For a standard out-of-source build, we create a separate ``build`` folder and run ``cmake``
in it:

.. code-block:: bash

  conda activate tarantella

  cd tarantella
  mkdir build && cd build
  export TARANTELLA_INSTALLATION_PATH=/your/installation/path
  cmake -DCMAKE_INSTALL_PREFIX=${TARANTELLA_INSTALLATION_PATH} \
        -DCMAKE_PREFIX_PATH=${GASPICXX_INSTALLATION_PATH} ../

Now, we can compile and install Tarantella to ``TARANTELLA_INSTALLATION_PATH``:

.. code-block:: bash

  make -j$(nproc) install
  export PATH=${TARANTELLA_INSTALLATION_PATH}/bin:${PATH}


[Optional] Building and running tests
-------------------------------------

In order to build Tarantella with tests, please follow the steps from the  
`Tarantella docs <https://tarantella.readthedocs.io/installation#optional-building-and-running-tests>`_.

