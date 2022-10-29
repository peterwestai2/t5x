'''

register eval tasks

'''


from t5.data import preprocessors
import t5
import seqio
import functools
import tensorflow as tf
import random
from t5.evaluation import metrics



vocabulary = t5.data.get_default_vocabulary()


DEFAULT_OUTPUT_FEATURES = {
    "inputs":
        seqio.Feature(
            vocabulary=vocabulary, add_eos=True),
    "targets":
        seqio.Feature(
            vocabulary=vocabulary, add_eos=True)
}




### 
# premise task
###
split_map = {'test':'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/tsv/v_2_2/eval_sets/test_eval_premise.tsv',
            'validation':'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/tsv/v_2_2/eval_sets/val_eval_premise.tsv'}

seqio.TaskRegistry.add(
    "atomic_eval_premise_v1",
    seqio.TextLineDataSource(split_map,skip_header_lines=1,),
    preprocessors=[
        
        functools.partial(
          t5.data.preprocessors.parse_tsv,
          #field_names=['head' ,'relation' ,'tail']),

          field_names=['head', 'relation', 'tail', 'split', 'rec_0.6', 'rec_0.9', 'rec_0.5', 'rec_0.7', 'rec_0.8', 'p_valid_model', 'inference', 'valid', 'readable_relation','readable_relation', 'inputs', 'targets']),
        seqio.preprocessors.tokenize, seqio.preprocessors.append_eos
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.accuracy,metrics.bleu])


### 
# hypothesis task
###
split_map = {'test':'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/tsv/v_2_2/eval_sets/test_eval_hypothesis.tsv',
            'validation':'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/tsv/v_2_2/eval_sets/val_eval_hypothesis.tsv'}

seqio.TaskRegistry.add(
    "atomic_eval_hypothesis_v1",
    seqio.TextLineDataSource(split_map,skip_header_lines=1,),
    preprocessors=[
        
        functools.partial(
          t5.data.preprocessors.parse_tsv,
          #field_names=['head' ,'relation' ,'tail']),

          field_names=['head', 'relation', 'tail', 'split', 'rec_0.6', 'rec_0.9', 'rec_0.5', 'rec_0.7', 'rec_0.8', 'p_valid_model', 'inference', 'valid', 'readable_relation','readable_relation', 'inputs', 'targets']),
        seqio.preprocessors.tokenize, seqio.preprocessors.append_eos
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.accuracy,metrics.bleu])


### 
# hypothesis task
###
split_map = {'test':'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/tsv/v_2_2/eval_sets/test_eval_plausibility.tsv',
            'validation':'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/tsv/v_2_2/eval_sets/validation_eval_plausibility.tsv'}

seqio.TaskRegistry.add(
    "atomic_eval_plausibility_v1",
    seqio.TextLineDataSource(split_map,skip_header_lines=1,),
    preprocessors=[
        
        functools.partial(
          t5.data.preprocessors.parse_tsv,
          #field_names=['head' ,'relation' ,'tail']),

          field_names=['head', 'relation', 'tail', 'split', 'rec_0.6', 'rec_0.9', 'rec_0.5', 'rec_0.7', 'rec_0.8', 'p_valid_model', 'inference', 'valid', 'readable_relation','readable_relation', 'inputs', 'targets']),
        seqio.preprocessors.tokenize, seqio.preprocessors.append_eos
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.accuracy,metrics.bleu])

seqio.MixtureRegistry.add(
  "atomic_eval_mix_v1",
  [("atomic_eval_plausibility_v1", 1), ("atomic_eval_hypothesis_v1", 1), ("atomic_eval_premise_v1",1)]
)
