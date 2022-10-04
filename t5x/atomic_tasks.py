import seqio
import functools
import tensorflow as tf
import t5
from t5.data import preprocessors
import random

'''

This file defines the seqio tasks to be used for the generator and critic tasks in 
COMET 2023

'''






vocabulary = t5.data.get_default_vocabulary()


DEFAULT_OUTPUT_FEATURES = {
    "inputs":
        seqio.Feature(
            vocabulary=vocabulary, add_eos=True),
    "targets":
        seqio.Feature(
            vocabulary=vocabulary, add_eos=True)
}



'''

First define atomic gen task

'''

split_map_gen = {'train':'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/tsv/v_2_1/train_gen.tsv',
            'test':'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/tsv/v_2_1/test_gen.tsv',
            'dev':'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/tsv/v_2_1/val_gen.tsv'}


def atomic_preprocessor_gen(ds):

  def to_inputs_and_targets(ex):
    src = tf.strings.join(['write a fact about this event: ', ex['head'] ])
    tgt = tf.strings.join([ex['readable_relation'], ex['inference']])
    
    return {
        'inputs':
             tf.strings.join(
                 [(src)]),
        'targets': 
        tf.strings.join(
                 [(tgt)]),
    }
  return ds.map(to_inputs_and_targets, 
                num_parallel_calls=tf.data.experimental.AUTOTUNE)


seqio.TaskRegistry.add(
    "atomic_gen",
    seqio.TextLineDataSource(split_map_gen,skip_header_lines=1,),
    preprocessors=[
        
        functools.partial(
          t5.data.preprocessors.parse_tsv,
          #field_names=['head' ,'relation' ,'tail']),

          field_names=['head' ,'relation' ,'tail' ,'split' ,'rec_0.6' ,'rec_0.9' ,'rec_0.5' ,'rec_0.7' ,'rec_0.8' ,'p_valid_model' ,'inference' ,'valid','readable_relation']),
        atomic_preprocessor_gen,
        seqio.preprocessors.tokenize, seqio.preprocessors.append_eos
    ],
    output_features=DEFAULT_OUTPUT_FEATURES)




'''

then define critic task

'''

split_map_critic = {'train':'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/tsv/v_2_1/train_critic.tsv',
            'test':'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/tsv/v_2_1/test_critic.tsv',
            'dev':'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/tsv/v_2_1/val_critic.tsv'}


def atomic_preprocessor_critic(ds):

  def to_inputs_and_targets(ex):
    src = tf.strings.join(['give a plausibility score: ', ex['head'],' ',ex['readable_relation'], ex['inference'] ])
    tgt = ex['valid']
    
    return {
        'inputs':
             tf.strings.join(
                 [(src)]),
        'targets': 
        tf.strings.join(
                 [(tgt)]),
    }
  return ds.map(to_inputs_and_targets, 
                num_parallel_calls=tf.data.experimental.AUTOTUNE)


seqio.TaskRegistry.add(
    "atomic_critic",
    seqio.TextLineDataSource(split_map_critic,skip_header_lines=1,),
    preprocessors=[
        
        functools.partial(
          t5.data.preprocessors.parse_tsv,
          #field_names=['head' ,'relation' ,'tail']),

          field_names=['head' ,'relation' ,'tail' ,'split' ,'rec_0.6' ,'rec_0.9' ,'rec_0.5' ,'rec_0.7' ,'rec_0.8' ,'p_valid_model' ,'inference' ,'valid','readable_relation']),
        atomic_preprocessor_critic,
        seqio.preprocessors.tokenize, seqio.preprocessors.append_eos
    ],
    output_features=DEFAULT_OUTPUT_FEATURES)


'''

Finally, define a mixture between the two tasks

'''

seqio.MixtureRegistry.add(
  "atomic_mix",
  [("atomic_critic", 1), ("atomic_gen", 10)]
)
