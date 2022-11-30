import seqio
import t5.data
import tensorflow as tf
import functools
from t5.evaluation import metrics
from t5.data import preprocessors

TaskRegistry = seqio.TaskRegistry

DEFAULT_OUTPUT_FEATURES = {
    "inputs": seqio.Feature(
        vocabulary=t5.data.get_default_vocabulary(), add_eos=True,
        required=False),
    "targets": seqio.Feature(
        vocabulary=t5.data.get_default_vocabulary(), add_eos=True)
}

keys_to_features = {
  'inputs': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
  'targets':tf.io.FixedLenFeature(shape=(), dtype=tf.string),
}

seqio.TaskRegistry.add(
    "diamond_pretraining_v1",
    seqio.TFExampleDataSource(
        {'train': 'gs://ai2-jack/diamond_finetune/training_data/train_100perc.tfrecord-*-of-0016',
         'validation': 'gs://ai2-jack/diamond_finetune/training_data/valid_100perc.tfrecord-*-of-0016',
         'test': 'gs://ai2-jack/diamond_finetune/training_data/test_100perc.tfrecord-*-of-0016'},
        keys_to_features,
    ),

    output_features = DEFAULT_OUTPUT_FEATURES,
    preprocessors=[
        seqio.preprocessors.tokenize,
        # seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos,
        functools.partial(
            preprocessors.rekey, key_map={key:key for key in DEFAULT_OUTPUT_FEATURES.keys()})
    ],
    metric_fns=[metrics.bleu]
)


seqio.TaskRegistry.add(
    "diamond_pretraining_v2",
    seqio.TFExampleDataSource(
        {'train': 'gs://ai2-jack/diamond_finetune/training_data_v2/train_100perc.tfrecord-*-of-0016',
         'validation': 'gs://ai2-jack/diamond_finetune/training_data_v2/valid_2perc.tfrecord-*-of-0016',
         'test': 'gs://ai2-jack/diamond_finetune/training_data_v2/test_2perc.tfrecord-*-of-0016'},
        keys_to_features,
    ),

    output_features = DEFAULT_OUTPUT_FEATURES,
    preprocessors=[
        seqio.preprocessors.tokenize,
        # seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos,
        functools.partial(
            preprocessors.rekey, key_map={key:key for key in DEFAULT_OUTPUT_FEATURES.keys()})
    ],
    metric_fns=[metrics.bleu]
)


seqio.TaskRegistry.add(
    "diamond_pretraining_v3",
    seqio.TFExampleDataSource(
        {'train': 'gs://ai2-jack/diamond_finetune/training_data_v3/train_100perc.tfrecord-*-of-0016',
         'validation': 'gs://ai2-jack/diamond_finetune/training_data_v3/valid_2perc.tfrecord-*-of-0016',
         'test': 'gs://ai2-jack/diamond_finetune/training_data_v3/test_2perc.tfrecord-*-of-0016'},
        keys_to_features,
    ),

    output_features = DEFAULT_OUTPUT_FEATURES,
    preprocessors=[
        seqio.preprocessors.tokenize,
        # seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos,
        functools.partial(
            preprocessors.rekey, key_map={key:key for key in DEFAULT_OUTPUT_FEATURES.keys()})
    ],
    metric_fns=[metrics.bleu]
)
