'''

The idea here is to use very basic tasks (just "inputs"/"targets") and 

'''

import seqio
import functools
import tensorflow as tf
import t5
from t5.data import preprocessors
import random


'''

Define variables used across tasks

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

march30_logic_task_v0

-- logit puzzle task from Yuchen around march 30

'''
#'peterw-tpu-eu/SoR/data/logic_v0/logic_grid_puzzles.dev.qa.tsv'
file_template = 'gs://peterw-tpu-eu/SoR/data/logic_v0/logic_grid_puzzles.{}.qa.tsv' #'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/jan_9_2022_dataset/unannotated_ATOMIC10X_{}.tsv'
input_files = {'train':file_template.format('train'),
            'test':file_template.format('test'),
            'validation':file_template.format('dev')}

seqio.TaskRegistry.add(
    "march31_logic_task",
    seqio.TextLineDataSource(input_files,skip_header_lines=1,),
    preprocessors=[
        
        functools.partial(
          t5.data.preprocessors.parse_tsv,
          #field_names=['head' ,'relation' ,'tail']),

          field_names=['inputs','targets']),
        seqio.preprocessors.tokenize, seqio.preprocessors.append_eos
    ],
    output_features=DEFAULT_OUTPUT_FEATURES)





      
      
      
'''

march31_math_task_v0

-- Nouha's task, the first version, sent on march31

'''
#'peterw-tpu-eu/SoR/data/math_v0/up_to_4_by_4_digit_dev.tsv'
file_template = 'gs://peterw-tpu-eu/SoR/data/math_v0/up_to_4_by_4_digit_{}.tsv'
input_files = {'train':file_template.format('train'),
            'test':file_template.format('test'),
            'validation':file_template.format('dev')}

seqio.TaskRegistry.add(
    "march31_math_task",
    seqio.TextLineDataSource(input_files,skip_header_lines=1,),
    preprocessors=[
        
        functools.partial(
          t5.data.preprocessors.parse_tsv,
          #field_names=['head' ,'relation' ,'tail']),

          field_names=['inputs','targets']),
        seqio.preprocessors.tokenize, seqio.preprocessors.append_eos
    ],
    output_features=DEFAULT_OUTPUT_FEATURES)




'''

April12_logic_task_v1

-- logit puzzle task from Yuchen April 12 (updated with new format and distribution)

'''

peterw-tpu-eu/SoR/data/logic_v1/logic_grid_puzzles.dev.qa.tsv
#'peterw-tpu-eu/SoR/data/logic_v0/logic_grid_puzzles.dev.qa.tsv'
file_template = 'gs://peterw-tpu-eu/SoR/data/logic_v1/logic_grid_puzzles.{}.qa.tsv' 
input_files = {'train':file_template.format('train'),
            'test':file_template.format('test'),
            'validation':file_template.format('dev')}

seqio.TaskRegistry.add(
    "April12_logic_task",
    seqio.TextLineDataSource(input_files,skip_header_lines=1,),
    preprocessors=[
        
        functools.partial(
          t5.data.preprocessors.parse_tsv,
          #field_names=['head' ,'relation' ,'tail']),

          field_names=['inputs','targets']),
        seqio.preprocessors.tokenize, seqio.preprocessors.append_eos
    ],
    output_features=DEFAULT_OUTPUT_FEATURES)
