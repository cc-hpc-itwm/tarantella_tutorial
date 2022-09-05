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
Before running the model, we need to add it to the existing ``PYTHONPATH``.

.. code-block:: bash

    export PYTHONPATH=${TNT_MODELS_PATH}:${PYTHONPATH}

Furthermore, the ``ImageNet`` dataset needs to be installed and available on
all the nodes that we want to use for training.
TensorFlow provides convenience scripts to download datasets, in their ``datasets``
package that is installed as a dependency for the TensorFlow Model Garden.
Install ImageNet to your local machine as described
`here <https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/scripts/download_and_prepare.py>`_.

.. code-block:: bash

    export TNT_DATASETS_PATH=/path/to/downloaded/datasets

    python -m tensorflow_datasets.scripts.download_and_prepare \
    --datasets=imagenet2012 --data_dir=${TNT_DATASETS_PATH}


Let's assume we have access to two nodes (saved in ``hostfile``) equipped with 4 GPUs each.
We can now simply run the ResNet-50 as follows:

.. code-block:: bash

    tarantella --hostfile ./hostfile --devices-per-node 4 \
    -- ${TNT_MODELS_PATH}/models/image_classification/train_imagenet_main.py --data_dir=${TNT_DATASETS_PATH} \
                                                                             --model_arch=resnet50 \
                                                                             --strategy=data \
                                                                             --batch_size=512 \
                                                                             --train_epochs=90 \
                                                                             --epochs_between_evals=10

The above command will train a ResNet-50 models on the 8 devices available in parallel
for ``90`` epochs, as suggested in [Goyal]_ to achieve convergence.
The ``--val_freq`` parameter specifies the frequency of evaluations of the
*validation dataset* performed in between training epochs.

Note the ``--batch_size`` parameter, which specifies the global batch size used in training.

Implementation overview
^^^^^^^^^^^^^^^^^^^^^^^
We will now look closer into the implementation of the ResNet-50 training scheme.
The main training steps reside in the ``models/image_classification/train_imagenet_main.py`` file.

The most important step in enabling data parallelism with Tarantella is
to wrap the Keras model into a Tarantella model that uses data parallelism for speeding up training.

This is summarized below for the `ResNet50` model:

.. code-block:: python

  model = tf.keras.applications.resnet50.ResNet50(include_top=True, weights=None, classes=1000,
                                                  input_shape=(224, 224, 3), input_tensor=None,
                                                  pooling=None, classifier_activation='softmax')
  ...
  if args.distribute == ParallelMethods.TNT:
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
<https://github.com/cc-hpc-itwm/tarantella_models/blob/master/src/models/image_classification/train_imagenet_main.py#L120>`_
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

If model checkpointing is required, it can be enabled through the ``ModelCheckpoint``
callback as usual (cf. :ref:`checkpointing models with Tarantella <checkpointing-via-callbacks-label>`).

.. code-block:: python

  callbacks.append(tf.keras.callbacks.ModelCheckpoint(ckpt_full_path, save_weights_only=True))


There is no need for any further changes to proceed with distributed training:

.. code-block:: python

  history = model.fit(train_dataset,
                      validation_data = val_dataset,
                      validation_freq=args.val_freq,
                      epochs=args.train_epochs,
                      callbacks=callbacks,
                      verbose=args.verbose)

Advanced topics
^^^^^^^^^^^^^^^

Scaling the batch size
""""""""""""""""""""""

Increasing the batch size provides a simple means to achieve significant training
time speed-ups, as it leads to perfect scaling with respect to the steps required
to achieve the target accuracy (up to some dataset- and model- dependent critical
size, after which further increasing the batch size only leads to diminishing returns)
[Shallue]_.

This observation, together with the fact that small local batch sizes decrease the
efficiency of DNN operators, represent the basis for a standard technique in data
parallelism: *using a fixed micro batch size and scaling the global batch size
with the number of devices*.

Tarantella provides multiple mechanisms to set the batch size, as presented in the
:ref:`Quick Start guide<using-distributed-datasets-label>`.

In the case of ResNet-50, we specify the global batch size as a command line
parameter, and let the framework divide the dataset into microbatches.

.. _scale-learning-rate-label:

Scaling the learning rate
"""""""""""""""""""""""""

To be able to reach the same target accuracy when scaling the global batch size up,
other hyperparameters need to be carefully tuned [Shallue]_.
In particular, adjusting the learning rate is essential for achieving convergence
at large batch sizes. [Goyal]_ proposes to *scale the
learning rate up linearly with the batch size* (and thus with the number of devices).

The scaled-up learning rate is set up at the begining of training, after which the
learning rate evolves over the training steps based on a so-called
*learning rate schedule*.

In our ResNet-50 example, we use a
`ExpDecayWithWarmupSchedule
<https://github.com/cc-hpc-itwm/tarantella_models/blob/master/src/models/image_classification/lr_scheduler.py#L83>`__.

Another type of schedule that we have implemented is the
`PiecewiseConstantDecayWithWarmup
<https://github.com/cc-hpc-itwm/tarantella_models/blob/master/src/models/image_classification/lr_scheduler.py#L10>`__
schedule, which is similar to the schedule introduced by [Goyal]_.

In both schedules, when training starts, the learning rate is initialized to
a large value that allows to explore more of the search space. The learning rate will
then decay the closer the algorithm gets to convergence.

The initial learning rate in the `ExpDecayWithWarmupSchedule` is scaled linearly with the
number of devices used as follows:

.. code-block:: bash

  initial_learning_rate = base_learning_rate * num_ranks

Learning rate warm-up
"""""""""""""""""""""

Whereas scaling up the learning rate with the batch size is necessary, a large learning
rate might degrade the stability of the optimization algorithm, especially in early training.
A technique to mitigate this limitation is to *warm-up* the learning rate during the first
epochs, particularly when using large batches [Goyal]_.

In our ResNet-50 example, the `ExpDecayWithWarmupSchedule` schedule
starts with a small value for the learning rate, which then increases at every step
(i.e., iteration), for a number of initial
`warmup_steps <https://github.com/cc-hpc-itwm/tarantella_models/blob/master/src/models/image_classification/lr_scheduler.py#L95>`_.

The ``warmup_steps`` value defaults to the number of iterations of the first five epochs,
matching the schedule proposed by [Goyal]_.
After the ``warmup_steps`` are done, the learning rate value should reach the *scaled initial
learning rate* introduced above.

.. code-block:: python

  def warmup():
    # Learning rate increases linearly per step.
    multiplier = self.warmup_rate * (step / self.warmup_steps)
    return tf.multiply(self.initial_learning_rate, multiplier)
