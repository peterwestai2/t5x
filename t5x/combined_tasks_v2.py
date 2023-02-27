'''

Backup of the updates I made to the new standard tasks, while I try to fix whatever broke it

'''


#from atomic_seqio_utils import key_prefixes, fill_missing_fields, mask_each_field, combine_inputs

from standard_utils_v2 import build_task

from t5.data import preprocessors
import t5
from t5.evaluation import metrics
import seqio
import functools
import tensorflow as tf
import random
import itertools

'''


Below, we define different task sets and mixtures based on these


'''


    
# ==================================== feb8 30k annotation dataset ======================================
# This creates the feb8 datasets, which use 30k annotation and the cqi format with plausibilities
# of -1.0, 0.0, 1.0, and 2.0
#

  
  
'''
Datasets -- ATOMIC-10X
'''

dataset_name = 'ATOMIC10X_feb8'
file_template = 'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/feb8_dataset/unannotated_ATOMIC10X_{}.tsv'
input_files = {'train':file_template.format('train'),
            'test':file_template.format('test'),
            'validation':file_template.format('val')}
mask_fields = ['context','query','inference']
build_task(input_files, dataset_name ,mask_fields, metric_fns =[metrics.bleu,metrics.rouge])

'''
Datasets -- ATOMIC2020 
'''

dataset_name = 'ATOMIC2020_feb8'
file_template = 'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/feb8_dataset/unannotated_ATOMIC2020_{}.tsv'
input_files = {'train':file_template.format('train'),
            'test':file_template.format('test'),
            'validation':file_template.format('val')}
mask_fields = ['context','query','inference']
build_task(input_files, dataset_name ,mask_fields, metric_fns =[metrics.bleu,metrics.rouge])


'''
Datasets -- generated 
'''

dataset_name = 'generated_feb8'
file_template = 'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/feb8_dataset/unannotated_generated2023_{}.tsv'
input_files = {'train':file_template.format('train'),
            'test':file_template.format('test'),
            'validation':file_template.format('val')}
mask_fields = ['context','query','inference']
build_task(input_files, dataset_name ,mask_fields, metric_fns =[metrics.bleu,metrics.rouge])







'''
Datasets -- human annotated

The annotated dataset, but also masking every annotation field and never
masking the generative fields.

'''

dataset_name = 'human_annotated_feb8'
file_template = 'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/feb8_dataset/annotated_full_{}.tsv'
input_files = {'train':file_template.format('train'),
            'test':file_template.format('test'),
            'validation':file_template.format('val')}
mask_fields =  ['context','query','inference','plausibility']
build_task(input_files, dataset_name ,mask_fields, metric_fns =[metrics.bleu,metrics.rouge])



'''
Datasets -- full human annotated critic

The annotated dataset, but also masking every annotation field and never
masking the generative fields. This multival version takes on values -1,0,1,2 instead of just 0,1

'''

dataset_name = 'full_critic_feb8'
file_template = 'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/feb8_dataset/annotated_full_{}.tsv'
input_files = {'train':file_template.format('train'),
            'test':file_template.format('test'),
            'validation':file_template.format('val')}
mask_fields =  ['plausibility']
build_task(input_files, dataset_name ,mask_fields, metric_fns =[metrics.bleu,metrics.rouge])


'''
Datasets -- round1 human annotated critic

The annotated dataset, but also masking every annotation field and never
masking the generative fields. This multival version takes on values -1,0,1,2 instead of just 0,1

'''

dataset_name = 'round1_critic_feb8'
# need a new one
file_template = 'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/feb8_dataset/annotated_round1_{}.tsv'
input_files = {'train':file_template.format('train'),
            'test':file_template.format('test'),
            'validation':file_template.format('val')}
mask_fields =  ['plausibility']
build_task(input_files, dataset_name ,mask_fields, metric_fns =[metrics.bleu,metrics.rouge])

'''
Datasets -- round2 human annotated critic

The annotated dataset, but also masking every annotation field and never
masking the generative fields. This multival version takes on values -1,0,1,2 instead of just 0,1

'''

dataset_name = 'round2_critic_feb8'
file_template = 'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/feb8_dataset/annotated_round2_{}.tsv'
input_files = {'train':file_template.format('train'),
            'test':file_template.format('test'),
            'validation':file_template.format('val')}
mask_fields =  ['plausibility']
build_task(input_files, dataset_name ,mask_fields, metric_fns =[metrics.bleu,metrics.rouge])



seqio.MixtureRegistry.add(
  "feb8_gen_mix",
  [('ATOMIC10X_feb8', 1),
  ('ATOMIC2020_feb8',1),
  ('generated_feb8',4),
  ('human_annotated_feb8',2),
  ('full_critic_feb8',6)])


seqio.MixtureRegistry.add(
  "feb8_critic_mix",
  [('full_critic_feb8',100),
  ('round1_critic_feb8',1),
  ('round2_critic_feb8',1)])





# ==================================== feb10 30k annotation dataset ======================================
# This creates the feb8 datasets, which use 30k annotation and the cqi format with plausibilities
# of 0, 1, 2, 3
#

  
  
'''
Datasets -- ATOMIC-10X
'''

dataset_name = 'ATOMIC10X_feb10'
file_template = 'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/feb10_dataset/unannotated_ATOMIC10X_{}.tsv'
input_files = {'train':file_template.format('train'),
            'test':file_template.format('test'),
            'validation':file_template.format('val')}
mask_fields = ['context','query','inference']
build_task(input_files, dataset_name ,mask_fields, metric_fns =[metrics.bleu,metrics.rouge])

'''
Datasets -- ATOMIC2020 
'''

dataset_name = 'ATOMIC2020_feb10'
file_template = 'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/feb10_dataset/unannotated_ATOMIC2020_{}.tsv'
input_files = {'train':file_template.format('train'),
            'test':file_template.format('test'),
            'validation':file_template.format('val')}
mask_fields = ['context','query','inference']
build_task(input_files, dataset_name ,mask_fields, metric_fns =[metrics.bleu,metrics.rouge])


'''
Datasets -- generated 
'''

dataset_name = 'generated_feb10'
file_template = 'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/feb10_dataset/unannotated_generated2023_{}.tsv'
input_files = {'train':file_template.format('train'),
            'test':file_template.format('test'),
            'validation':file_template.format('val')}
mask_fields = ['context','query','inference']
build_task(input_files, dataset_name ,mask_fields, metric_fns =[metrics.bleu,metrics.rouge])







'''
Datasets -- human annotated

The annotated dataset, but also masking every annotation field and never
masking the generative fields.

'''

dataset_name = 'human_annotated_feb10'
file_template = 'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/feb10_dataset/annotated_full_{}.tsv'
input_files = {'train':file_template.format('train'),
            'test':file_template.format('test'),
            'validation':file_template.format('val')}
mask_fields =  ['context','query','inference','plausibility']
build_task(input_files, dataset_name ,mask_fields, metric_fns =[metrics.bleu,metrics.rouge])



'''
Datasets -- full human annotated critic


'''

dataset_name = 'full_critic_feb10'
file_template = 'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/feb10_dataset/annotated_full_{}.tsv'
input_files = {'train':file_template.format('train'),
            'test':file_template.format('test'),
            'validation':file_template.format('val')}
mask_fields =  ['plausibility']
build_task(input_files, dataset_name ,mask_fields, metric_fns =[metrics.bleu,metrics.rouge])


'''
Datasets -- round1 human annotated critic

'''

dataset_name = 'round1_critic_feb10'
# need a new one
file_template = 'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/feb10_dataset/annotated_round1_{}.tsv'
input_files = {'train':file_template.format('train'),
            'test':file_template.format('test'),
            'validation':file_template.format('val')}
mask_fields =  ['plausibility']
build_task(input_files, dataset_name ,mask_fields, metric_fns =[metrics.bleu,metrics.rouge])

'''
Datasets -- round2 human annotated critic


'''

dataset_name = 'round2_critic_feb10'
file_template = 'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/feb10_dataset/annotated_round2_{}.tsv'
input_files = {'train':file_template.format('train'),
            'test':file_template.format('test'),
            'validation':file_template.format('val')}
mask_fields =  ['plausibility']
build_task(input_files, dataset_name ,mask_fields, metric_fns =[metrics.bleu,metrics.rouge])



seqio.MixtureRegistry.add(
  "feb10_gen_mix",
  [('ATOMIC10X_feb10', 1),
  ('ATOMIC2020_feb10',1),
  ('generated_feb10',4),
  ('human_annotated_feb10',2),
  ('full_critic_feb10',6)])


seqio.MixtureRegistry.add(
  "feb10_critic_mix",
  [('full_critic_feb10',100),
  ('round1_critic_feb10',1),
  ('round2_critic_feb10',1)])



'''


New mix that only uses the generated data 
for the main bulk of training

'''

seqio.MixtureRegistry.add(
  "feb24_gen_mix",
  [('generated_feb8',4),
  ('human_annotated_feb8',2),
  ('full_critic_feb8',6)])





# ==================================== feb27 ablation ======================================
# This creates the tasks for the critic ablation on feb27 using the 20k critic set
#

  
 

'''
Datasets -- critic ablations
'''


for name, _ in ([('1k',1000), ('2k',2000),('4k',4000),('8k',8000),('16k',16000)]):
    dataset_name = 'feb27_ablation_{}'.format(name)
    file_template = 'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/feb27_ablation/annotated_{}_'.format(name) + '{}.tsv'
    input_files = {'train':file_template.format('train'),
                'test':file_template.format('test'),
                'validation':file_template.format('val')}
    mask_fields =  ['plausibility']
    build_task(input_files, dataset_name ,mask_fields, metric_fns =[metrics.bleu,metrics.rouge])


