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

#peterw-tpu-eu/SoR/data/logic_v1/logic_grid_puzzles.dev.qa.tsv
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

'''

april19_math_scratchpad_upto_4_by_1

-- Nouha's task, second version

'''


#'peterw-tpu-eu/SoR/data/math_v0/up_to_4_by_4_digit_dev.tsv'
file_template = 'gs://peterw-tpu-eu/SoR/data/april19_math_scratchpad_upto_4_by_1/scratchpad_upto_4_by_1_prompts_{}.tsv'
input_files = {'train':file_template.format('train'),
            'test':file_template.format('test'),
            'validation':file_template.format('dev')}

seqio.TaskRegistry.add(
    "april19_math_scratchpad_upto_4_by_1",
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

may 1 math tasks

'''

# max i/o lengths are 10 / 537
file_template = 'gs://peterw-tpu-eu/SoR/data/may1_math_tasks/scratchpad_3_by_2_prompts_tot_{}.tsv'
input_files = {'train':file_template.format('train'),
            'test':file_template.format('test'),
            'validation':file_template.format('dev')}

seqio.TaskRegistry.add(
    "may1_scratchpad_3_by_2_prompts_tot",
    seqio.TextLineDataSource(input_files,skip_header_lines=1,),
    preprocessors=[
        
        functools.partial(
          t5.data.preprocessors.parse_tsv,
          #field_names=['head' ,'relation' ,'tail']),

          field_names=['inputs','targets']),
        seqio.preprocessors.tokenize, seqio.preprocessors.append_eos
    ],
    output_features=DEFAULT_OUTPUT_FEATURES)


# max i/o lengths are 10 / 5 (so maybe make both 16?)
file_template = 'gs://peterw-tpu-eu/SoR/data/may1_math_tasks/up4_by_2_digit_fine_tune_{}.tsv'
input_files = {'train':file_template.format('train'),
            'test':file_template.format('test'),
            'validation':file_template.format('dev')}

seqio.TaskRegistry.add(
    "may1_up4_by_2_digit_fine_tune",
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

may 1 logic tasks

'''

# max i/o lengths are 367 / 142
file_template = 'gs://peterw-tpu-eu/SoR/data/may1_logic_tasks/lgp.{}.tsv'
input_files = {'train':file_template.format('train'),
            'test':file_template.format('test'),
            'validation':file_template.format('dev')}

seqio.TaskRegistry.add(
    "may1_lgp",
    seqio.TextLineDataSource(input_files,skip_header_lines=1,),
    preprocessors=[
        
        functools.partial(
          t5.data.preprocessors.parse_tsv,
          #field_names=['head' ,'relation' ,'tail']),

          field_names=['inputs','targets']),
        seqio.preprocessors.tokenize, seqio.preprocessors.append_eos
    ],
    output_features=DEFAULT_OUTPUT_FEATURES)


# max i/o lengths are 367 / 692
file_template = 'gs://peterw-tpu-eu/SoR/data/may1_logic_tasks/lgp.{}.tsv'
input_files = {'train':'gs://peterw-tpu-eu/SoR/data/may1_logic_tasks/lgp.train_reason.tsv',
            'test':file_template.format('test'),
            'validation':file_template.format('dev')}

seqio.TaskRegistry.add(
    "may1_lgp_with_reason",
    seqio.TextLineDataSource(input_files,skip_header_lines=1,),
    preprocessors=[
        
        functools.partial(
          t5.data.preprocessors.parse_tsv,
          #field_names=['head' ,'relation' ,'tail']),

          field_names=['inputs','targets']),
        seqio.preprocessors.tokenize, seqio.preprocessors.append_eos
    ],
    output_features=DEFAULT_OUTPUT_FEATURES)





# ==================================== may26 taylor task======================================
#  the basic dataset from taylor (situation -> gpt4 output)


file_template = 'gs://ai2-mosaic-private/peter-skd-2023/data/taylor_data/may26_basic/{}.tsv'
input_files = {'train':file_template.format('train'),
            'test':file_template.format('test'),
            'validation':file_template.format('dev')}

seqio.TaskRegistry.add(
    "may26_basic_taylor",
    seqio.TextLineDataSource(input_files,skip_header_lines=1,),
    preprocessors=[
        
        functools.partial(
          t5.data.preprocessors.parse_tsv,
          #field_names=['head' ,'relation' ,'tail']),

          field_names=['inputs','targets']),
        seqio.preprocessors.tokenize, seqio.preprocessors.append_eos
    ],
    output_features=DEFAULT_OUTPUT_FEATURES)

# ==================================== june1======================================
#  slightly more formatted data from Taylor: going from situation to a single value rather than full output


file_template = 'gs://ai2-mosaic-private/peter-skd-2023/data/taylor_data/june1_single/{}.tsv'
input_files = {'train':file_template.format('train'),
            'test':file_template.format('test'),
            'validation':file_template.format('val')}

seqio.TaskRegistry.add(
    "june1_single",
    seqio.TextLineDataSource(input_files,skip_header_lines=1,),
    preprocessors=[
        
        functools.partial(
          t5.data.preprocessors.parse_tsv,
          #field_names=['head' ,'relation' ,'tail']),

          field_names=['inputs','targets']),
        seqio.preprocessors.tokenize, seqio.preprocessors.append_eos
    ],
    output_features=DEFAULT_OUTPUT_FEATURES)



# ==================================== delphi data======================================
#  the delphi data to compare as pretraining

file_template = 'gs://ai2-mosaic-private/peter-skd-2023/data/taylor_data/delphi_data/commonsense-norm-bank/freeform/{}.freeform.tsv'
input_files = {'train':file_template.format('train'),
            'test':file_template.format('test'),
            'validation':file_template.format('validation')}

seqio.TaskRegistry.add(
    "june3_delphi_freeform",
    seqio.TextLineDataSource(input_files,skip_header_lines=1,),
    preprocessors=[
        
        functools.partial(
          t5.data.preprocessors.parse_tsv,
          #field_names=['head' ,'relation' ,'tail']),

          field_names=['index','inputs','class_label','targets','input_type','pattern','source']),
        seqio.preprocessors.tokenize, seqio.preprocessors.append_eos
    ],
    output_features=DEFAULT_OUTPUT_FEATURES)




# ==================================== aug7 grokking ======================================
# 3by2 data for grokking experiments

file_template = 'gs://ai2-mosaic-private/peterw_SoR/data/aug7_3by2/3by2_{}.tsv'
input_files = {'train':file_template.format('train'),
            'test':file_template.format('test'),
            'validation':file_template.format('dev')}

seqio.TaskRegistry.add(
    "aug7_3by2",
    seqio.TextLineDataSource(input_files,skip_header_lines=1,),
    preprocessors=[
        
        functools.partial(
          t5.data.preprocessors.parse_tsv,
          #field_names=['head' ,'relation' ,'tail']),

          field_names=['inputs','targets']),
        seqio.preprocessors.tokenize, seqio.preprocessors.append_eos
    ],
    output_features=DEFAULT_OUTPUT_FEATURES)
