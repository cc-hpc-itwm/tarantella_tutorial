.. image:: pics/tnt_logo_text.png
   :width: 750
   :align: center

|
`Tarantella <https://github.com/cc-hpc-itwm/tarantella>`_
is an open-source, distributed Deep Learning framework built on top of TensorFlow,
providing scalable Deep Neural Network training on CPU and GPU compute clusters.

Tarantella is easy-to-use, allows to re-use existing TensorFlow models,
and does not require any knowledge of parallel computing.

Distributed training in Tarantella is based on the simplest and most efficient parallelization
strategy for deep neural networks (DNNs), which is called *data parallelism*.

This strategy is already in use when deploying batched optimizers, such as stochastic
gradient descent (SGD) or ADAM. In this case, input samples are grouped together in 
so-called mini-batches and are processed in parallel.

Tarantella extends this scheme by splitting each mini-batch into a number of micro-batches,
which are then executed on different devices (e.g., GPUs).
In order to do this, the DNN is replicated on each device,
which then processes part of the data independently of the other devices.
During the *backpropagation* pass, partial results need to be accumulated via a so-called
`allreduce <https://en.wikipedia.org/wiki/Collective_operation#All-Reduce_%5B5%5D>`_
collective operation.


Table of contents
=================

.. toctree::
   :maxdepth: 2

   installation
   quick_start
   tutorials
   advanced_topics
   faq

