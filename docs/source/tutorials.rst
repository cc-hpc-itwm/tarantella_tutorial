.. _tutorials-label:

Image Classification with Tarantella
====================================

This section delves into a more advanced usage of Tarantella by looking at 
distributed training for state-of-the-art image classification models.

The image classification model architectures are imported through the
`tf.keras.applications <https://www.tensorflow.org/api_docs/python/tf/keras/applications>`_
module, available in recent TensorFlow releases.


ResNet-50
---------

Deep Residual Networks (ResNets) represented a breakthrough in the field of
computer vision, enabling deeper and more complex deep convolutional networks.
Introduced in [He]_, ResNet-50 has become a standard model for image classification
tasks, and has been shown to scale to very large number of nodes in data parallel
training [Goyal]_.

Run Resnet-50 with Tarantella
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let's assume we have access to two nodes (saved in ``hostfile``) equipped with 4 GPUs each.
We can now simply run the ResNet-50 as follows:

.. code-block:: bash

    cd examples/image_classification
    tarantella --hostfile ./hostfile --devices-per-node 4 \
    -- ./train_imagenet_main.py --model_arch=resnet50 --batch_size=256 --train_epochs=3 --val_freq=3
                                --train_num_samples=2560 --val_num_samples=256 --synthetic_data

The above command will train a ResNet-50 models on the 8 devices available in parallel
for ``3`` epochs, as suggested in [Goyal]_ to achieve convergence.
The ``--val_freq`` parameter specifies the frequency of evaluations of the
*validation dataset* performed in between training epochs.

Note the ``--batch_size`` parameter, which specifies the global batch size used in training.

The ``--synthetic_data`` instructs the code to generate a synthetic ImageNet-like dataset, that can be used
to showcase the training procedure. However, it will not provide any meaningful results.
Remove the ``--synthetic_data`` parameter a specify a ``--data_dir`` path to an actual ImageNet directory
to properly train the model.

.. note::

  On the STYX cluster, a pre-downloaded version of the ImageNet dataset can be found in ``/home/DATA/ImageNet``.
  

Implementation overview
^^^^^^^^^^^^^^^^^^^^^^^
We will now look closer into the implementation of the ResNet-50 training scheme.
The main training steps reside in the ``examples/image_classification/train_imagenet_main.py`` file.

The most important step in enabling data parallelism with Tarantella is
to wrap the Keras model into a Tarantella model that uses data parallelism for speeding up training.

This is summarized below for the `ResNet50` model:

.. code-block:: python

  model = tf.keras.applications.resnet50.ResNet50(include_top=True, weights=None, classes=1000,
                                                  input_shape=(224, 224, 3), input_tensor=None,
                                                  pooling=None, classifier_activation='softmax')
  model = tnt.Model(model,
                    parallel_strategy = tnt.ParallelStrategy.DATA)

Next, the training procedure can simply be written down as it would be for a
standard, TensorFlow-only model. No further changes are required to train the
model in a distributed manner.

In particular, the ImageNet dataset is loaded and preprocessed as follows:

.. code-block:: python

  train_input_dataset = load_dataset(dataset_type='train',
                                     data_dir=args.data_dir, num_samples = args.train_num_samples,
                                     batch_size=args.batch_size, dtype=tf.float32,
                                     drop_remainder=args.drop_remainder,
                                     shuffle_seed=args.shuffle_seed)

The
`load_dataset
<https://github.com/cc-hpc-itwm/tarantella_tutorial/blob/main/examples/image_classification/train_imagenet_main.py#L76>`_
function reads the input files in ``data_dir``, loads the training samples, and processes
them into TensorFlow datasets.

The user only needs to pass the global ``batch_size`` value, and the Tarantella
framework will ensure that the dataset is properly distributed among devices,
such that:

  * each device will process an independent set of samples
  * each device will group the samples into micro batches, where the micro-batch
    size will be roughly equal to ``batch_size / num_devices``.
    If the batch size is not a multiple of the number of ranks, the remainder samples
    will be equally distributed among the participating ranks, such that some ranks
    will use a micro-batch of ``(batch_size / num_devices) + 1``.
  * each device will apply the same set of transformations to its input samples as
    specified in the ``load_dataset`` function.

The advantage of using the *automatic dataset distribution* mechanism of Tarantella
is that users can reason about their I/O pipeline without taking care of the details
about how to distribute it.

Before starting the training, the model is compiled using a standard Keras optimizer
and loss.

.. code-block:: python

  model.compile('optimizer' : tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9),
                'loss' : tf.keras.losses.SparseCategoricalCrossentropy(),
                'metrics' : [tf.keras.metrics.SparseCategoricalAccuracy()])

We provide flags to enable the most commonly used Keras ``callbacks``, such as
the ``TensorBoard`` profiler, which can simply be passed to the ``fit`` function
of the Tarantella model.

.. code-block:: python

  callbacks.append(tf.keras.callbacks.TensorBoard(log_dir = flags_obj.model_dir,
                                                  profile_batch = 2))


There is no need for any further changes to proceed with distributed training:

.. code-block:: python

  history = model.fit(train_dataset,
                      validation_data = val_dataset,
                      validation_freq=args.val_freq,
                      epochs=args.train_epochs,
                      callbacks=callbacks,
                      verbose=args.verbose)


.. rubric:: References

.. [Goyal] Goyal, Priya, et al. "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour."
           `arXiv preprint arXiv:1706.02677 <https://arxiv.org/abs/1706.02677>`_ (2017).

.. [He] He, Kaiming, et al. "Deep residual learning for image recognition."
        Proceedings of the IEEE conference on computer vision and pattern recognition.
        `arXiv preprint arXiv:1512.03385 <https://arxiv.org/abs/1512.03385>`_ (2016).
