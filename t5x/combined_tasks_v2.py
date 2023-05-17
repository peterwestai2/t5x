'''

Backup of the updates I made to the new standard tasks, while I try to fix whatever broke it

'''


#from atomic_seqio_utils import key_prefixes, fill_missing_fields, mask_each_field, combine_inputs

from standard_utils_v2 import build_task, vocabulary

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



dataset_name = 'human_annotated_round2_feb10'
file_template = 'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/feb10_dataset/annotated_round2_{}.tsv'
input_files = {'train':file_template.format('train'),
            'test':file_template.format('test'),
            'validation':file_template.format('val')}
mask_fields =  ['context','query','inference','plausibility']
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

    
    
# ==================================== feb28 data_source_comparison ======================================
# This creates mixtures for the model with and without ATOMIC10X/2020 as we may drop these
#

seqio.MixtureRegistry.add(
  "feb28_gen_allsets_mix",
  [('ATOMIC10X_feb10', 1),
  ('ATOMIC2020_feb10',1),
  ('generated_feb10',4),
  ('human_annotated_round2_feb10',2),
  ('round2_critic_feb10',6)])

seqio.MixtureRegistry.add(
  "feb28_gen_newgen_mix",
  [('generated_feb10',6),
  ('human_annotated_round2_feb10',2),
  ('round2_critic_feb10',6)])



# ==================================== march 14 new data critic ======================================
# Using only generated data for the critic (both normal generations and some from curie for weaker)
#



dataset_name = 'march14_critic_strongweak_gen'
file_template = 'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/march14_critic_data/march14_critic_{}.tsv'
input_files = {'train':file_template.format('train'),
            'test':file_template.format('test'),
            'validation':file_template.format('val')}
mask_fields =  ['plausibility']
build_task(input_files, dataset_name ,mask_fields, metric_fns =[metrics.bleu,metrics.rouge])







# ==================================== april 2 data ======================================
# new data from taylor, with and without queries
#

dataset_name = 'april_2_gpt3turbo_v1'
file_template = 'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/april_2_data/round2-gpt-3.5-turbo_v1_{}.csv'
input_files = {'train':file_template.format('train'),
            'test':file_template.format('test'),
            'validation':file_template.format('val')}
mask_fields = ['context','inference']
build_task(input_files, dataset_name ,mask_fields, metric_fns =[metrics.bleu,metrics.rouge])


dataset_name = 'april_2_gpt3turbo_qa'
file_template = 'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/april_2_data/round2-gpt-3.5-turbo_qa_{}.csv'
input_files = {'train':file_template.format('train'),
            'test':file_template.format('test'),
            'validation':file_template.format('val')}
mask_fields = ['context','query','inference']
build_task(input_files, dataset_name ,mask_fields, metric_fns =[metrics.bleu,metrics.rouge])


seqio.MixtureRegistry.add(
  "april_2_gpt3turbo",
  [('april_2_gpt3turbo_v1',1),
  ('april_2_gpt3turbo_qa',1)])




# ==================================== april 3 critic ======================================
# frequency critic
#

'''
New critic using Jena's template

'''

dataset_name = 'april3_freq_critic'
file_template = 'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/april3_freq_critic/april3_freq_critic_{}.tsv'
input_files = {'train':file_template.format('train'),
            'test':file_template.format('test'),
            'validation':file_template.format('val')}
mask_fields =  ['plausibility']
build_task(input_files, dataset_name ,mask_fields, metric_fns =[metrics.bleu,metrics.rouge])




# ==================================== april 7 model_annoations ======================================
# first datasets using model annotation -- gpt3.5_turbo and gpt4, both zs
#


dataset_name = 'april7_model_annotation_turbo'
file_template = 'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/april7_model_annotations/april7_model_annotated_gpt3_turbo_zs_round1_20k_{}.tsv'
input_files = {'train':file_template.format('train'),
            'test':file_template.format('test'),
            'validation':file_template.format('val')}
mask_fields =  ['plausibility']
build_task(input_files, dataset_name ,mask_fields, metric_fns =[metrics.bleu,metrics.rouge])


dataset_name = 'april7_model_annotation_gpt4'
file_template = 'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/april7_model_annotations/april7_model_annotated_gpt4_zs_round1_20k_{}.tsv'
input_files = {'train':file_template.format('train'),
            'test':file_template.format('test'),
            'validation':file_template.format('val')}
mask_fields =  ['plausibility']
build_task(input_files, dataset_name ,mask_fields, metric_fns =[metrics.bleu,metrics.rouge])




## and then the fs versions

dataset_name = 'april11_model_annotation_fs_turbo'
file_template = 'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/april7_model_annotations/april11_model_annotated_gpt3_turbo_fs_round1_20k_{}.tsv'
input_files = {'train':file_template.format('train'),
            'test':file_template.format('test'),
            'validation':file_template.format('val')}
mask_fields =  ['plausibility']
build_task(input_files, dataset_name ,mask_fields, metric_fns =[metrics.bleu,metrics.rouge])


dataset_name = 'april11_model_annotation_fs_gpt4'
file_template = 'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/april7_model_annotations/april11_model_annotated_gpt4_fs_round1_20k_{}.tsv'
input_files = {'train':file_template.format('train'),
            'test':file_template.format('test'),
            'validation':file_template.format('val')}
mask_fields =  ['plausibility']
build_task(input_files, dataset_name ,mask_fields, metric_fns =[metrics.bleu,metrics.rouge])



# ==================================== april 14 to annotate ======================================
# the full dataset, with every possible label. This allows us to score labels for annotation
#


dataset_name = 'april14_to_annotate'
file_template = 'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/april14_to_annotate/april14_toannotate_{}.tsv'
input_files = {'train':file_template.format('train'),
            'test':file_template.format('test'),
            'validation':file_template.format('val')}
mask_fields =  ['plausibility']
build_task(input_files, dataset_name ,mask_fields, metric_fns =[metrics.bleu,metrics.rouge], tsv_fields=['context','query','inference','plausibility'])


dataset_name = 'april16_to_annotate'
file_template = 'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/april14_to_annotate/april16_toannotate_{}.tsv'
input_files = {'train':file_template.format('train'),
            'test':file_template.format('test'),
            'validation':file_template.format('val')}
mask_fields =  ['plausibility']
build_task(input_files, dataset_name ,mask_fields, metric_fns =[metrics.bleu,metrics.rouge], tsv_fields=['index','split','label','context','query','inference','plausibility'])



# ==================================== april 17 critic annotated ======================================
# data annotated by the round2 critic
#


dataset_name = 'april17_critic_annotated_query'
file_template = 'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/april17_critic_annotated/april17_criticannotated_query_{}.tsv'
input_files = {'train':file_template.format('train'),
            'test':file_template.format('test'),
            'validation':file_template.format('val')}
mask_fields =  ['context','query','inference','plausibility']
build_task(input_files, dataset_name ,mask_fields, metric_fns =[metrics.bleu,metrics.rouge])


dataset_name = 'april17_critic_annotated_noquery'
file_template = 'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/april17_critic_annotated/april17_criticannotated_noquery_{}.tsv'
input_files = {'train':file_template.format('train'),
            'test':file_template.format('test'),
            'validation':file_template.format('val')}
mask_fields =  ['context','inference','plausibility']
build_task(input_files, dataset_name ,mask_fields, metric_fns =[metrics.bleu,metrics.rouge])


seqio.MixtureRegistry.add(
  "april17_critic_annotated",
  [('april17_critic_annotated_query',1),
  ('april17_critic_annotated_noquery',1)])




# ==================================== april 24 ablation ======================================
# This creates the tasks for the model-annotation ablation (annotations by turbo)
#

  
 

'''
Datasets -- critic ablations
'''

for name, _ in ([('1k',1000), ('2k',2000),('5k',5000),('10k',10000),('20k',20000)]):
    dataset_name = 'april24_ablation_{}'.format(name)
    file_template = 'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/april21_model_annotation_ablation/april11_model_annotated_gpt3_turbo_fs_round1_{}' + '_{}.tsv'.format(name)
    input_files = {'train':file_template.format('train'),
                'test':file_template.format('test'),
                'validation':file_template.format('val')}
    mask_fields =  ['plausibility']
    build_task(input_files, dataset_name ,mask_fields, metric_fns =[metrics.bleu,metrics.rouge])
    
    
# ==================================== april 25 turbo annotations ======================================
# The 2 100k-sized datasets that Taylor sent over
#


dataset_name = 'april25_turbo_annotations_v1'
file_template = 'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/april25_turbo_annotations_100k/april25_turbo_annotations_v1_{}.tsv'
input_files = {'train':file_template.format('train'),
            'test':file_template.format('test'),
            'validation':file_template.format('val')}
mask_fields =  ['plausibility']
build_task(input_files, dataset_name ,mask_fields, metric_fns =[metrics.bleu,metrics.rouge], tsv_fields=['context','query','inference','plausibility'])



dataset_name = 'april25_turbo_annotations_v2'
file_template = 'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/april25_turbo_annotations_100k/april25_turbo_annotations_v2_{}.tsv'
input_files = {'train':file_template.format('train'),
            'test':file_template.format('test'),
            'validation':file_template.format('val')}
mask_fields =  ['plausibility']
build_task(input_files, dataset_name ,mask_fields, metric_fns =[metrics.bleu,metrics.rouge], tsv_fields=['context','query','inference','plausibility'])


# ==================================== april 28 combined ======================================
# combined generation/discrimination task (combining full turbo generations and turbo annotations)
# we can think of this as round0 because it's before we've done any generation

seqio.MixtureRegistry.add(
  "april28_gen_discr_round0",
  [('april_2_gpt3turbo_v1',1),
   ('april_2_gpt3turbo_qa',1),
   ('april25_turbo_annotations_v1',1)])




# ==================================== april 28 iterative ======================================
# multi-round iterative training tasks
#
#


dataset_name = 'april_28_round1_train_v1'
file_template = 'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/iterative/april28_data/round1/train_dataset_v1_{}.tsv'
input_files = {'train':file_template.format('train'),
            'test':file_template.format('test'),
            'validation':file_template.format('val')}
mask_fields = ['context','inference']
build_task(input_files, dataset_name ,mask_fields, metric_fns =[metrics.bleu,metrics.rouge])


dataset_name = 'april_28_round1_train_qa'
file_template = 'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/iterative/april28_data/round1/train_dataset_qa_{}.tsv'
input_files = {'train':file_template.format('train'),
            'test':file_template.format('test'),
            'validation':file_template.format('val')}
mask_fields = ['context','query','inference']
build_task(input_files, dataset_name ,mask_fields, metric_fns =[metrics.bleu,metrics.rouge])


seqio.MixtureRegistry.add(
  "april_28_round1_train",
  [('april_28_round1_train_v1',1),
  ('april_28_round1_train_qa',1),
  ('april25_turbo_annotations_v1',1)])





# ==================================== may6 iterative ======================================
# multi-round iterative training tasks
#
#


dataset_name = 'may6_train_round0_v1'
file_template = 'gs://ai2-mosaic-private/peter-skd-2023/iterative_runs/may6/data/round0/train_dataset_v1_{}.tsv'
input_files = {'train':file_template.format('train'),
            'test':file_template.format('test'),
            'validation':file_template.format('val')}
mask_fields = ['context','inference']
build_task(input_files, dataset_name ,mask_fields, metric_fns =[metrics.bleu,metrics.rouge])


dataset_name = 'may6_train_round0_qa'
file_template = 'gs://ai2-mosaic-private/peter-skd-2023/iterative_runs/may6/data/round0/train_dataset_qa_{}.tsv'
input_files = {'train':file_template.format('train'),
            'test':file_template.format('test'),
            'validation':file_template.format('val')}
mask_fields = ['context','query','inference']
build_task(input_files, dataset_name ,mask_fields, metric_fns =[metrics.bleu,metrics.rouge])


seqio.MixtureRegistry.add(
  "may6_train_round0",
  [('may6_train_round0_v1',1),
  ('may6_train_round0_qa',1),
  ('april25_turbo_annotations_v1',1)])



dataset_name = 'may6_train_round0_annotation'
file_template = 'gs://ai2-mosaic-private/peter-skd-2023/iterative_runs/may6/data/round0/data_to_score.tsv'
input_files = {'train':file_template,
            'test':file_template,
            'validation':file_template}
mask_fields =  ['plausibility']
build_task(input_files, dataset_name ,mask_fields, metric_fns =[metrics.bleu,metrics.rouge], tsv_fields=['context','query','inference','plausibility','split','generation_round','plausibility_p','index','label'])


dataset_name = 'may6_train_round1_v1'
file_template = 'gs://ai2-mosaic-private/peter-skd-2023/iterative_runs/may6/data/round1/train_dataset_v1_{}.tsv'
input_files = {'train':file_template.format('train'),
            'test':file_template.format('test'),
            'validation':file_template.format('val')}
mask_fields = ['context','inference']
build_task(input_files, dataset_name ,mask_fields, metric_fns =[metrics.bleu,metrics.rouge])


dataset_name = 'may6_train_round1_qa'
file_template = 'gs://ai2-mosaic-private/peter-skd-2023/iterative_runs/may6/data/round1/train_dataset_qa_{}.tsv'
input_files = {'train':file_template.format('train'),
            'test':file_template.format('test'),
            'validation':file_template.format('val')}
mask_fields = ['context','query','inference']
build_task(input_files, dataset_name ,mask_fields, metric_fns =[metrics.bleu,metrics.rouge])


seqio.MixtureRegistry.add(
  "may6_train_round1",
  [('may6_train_round1_v1',1),
  ('may6_train_round1_qa',1),
  ('april25_turbo_annotations_v1',1)])




DEFAULT_OUTPUT_FEATURES = {
    "inputs":
        seqio.Feature(
            vocabulary=vocabulary, add_eos=True),
    "targets":
        seqio.Feature(
            vocabulary=vocabulary, add_eos=True)
}

file_template = 'gs://ai2-mosaic-private/peter-skd-2023/iterative_runs/may6/data/round1/inference_dataset.tsv'

input_files = {'train':file_template,
            'test':file_template,
            'validation':file_template}

seqio.TaskRegistry.add(
    "may6_generate_round1",
    seqio.TextLineDataSource(input_files,skip_header_lines=1,),
    preprocessors=[
        
        functools.partial(
          t5.data.preprocessors.parse_tsv,
          #field_names=['head' ,'relation' ,'tail']),

          field_names=['context','query','inference','plausibility','split','generation_round','plausibility_p','inputs','targets']),
        seqio.preprocessors.tokenize, seqio.preprocessors.append_eos
    ],
    output_features=DEFAULT_OUTPUT_FEATURES)


dataset_name = 'may6_train_round1_annotation'
file_template = 'gs://ai2-mosaic-private/peter-skd-2023/iterative_runs/may6/data/round1/data_to_score.tsv'
input_files = {'train':file_template,
            'test':file_template,
            'validation':file_template}
mask_fields =  ['plausibility']
build_task(input_files, dataset_name ,mask_fields, metric_fns =[metrics.bleu,metrics.rouge], tsv_fields=['context','query','inference','plausibility','split','generation_round','plausibility_p','index','label'])


dataset_name = 'may6_train_round2_v1'
file_template = 'gs://ai2-mosaic-private/peter-skd-2023/iterative_runs/may6/data/round2/train_dataset_v1_{}.tsv'
input_files = {'train':file_template.format('train'),
            'test':file_template.format('test'),
            'validation':file_template.format('val')}
mask_fields = ['context','inference']
build_task(input_files, dataset_name ,mask_fields, metric_fns =[metrics.bleu,metrics.rouge])


dataset_name = 'may6_train_round2_qa'
file_template = 'gs://ai2-mosaic-private/peter-skd-2023/iterative_runs/may6/data/round2/train_dataset_qa_{}.tsv'
input_files = {'train':file_template.format('train'),
            'test':file_template.format('test'),
            'validation':file_template.format('val')}
mask_fields = ['context','query','inference']
build_task(input_files, dataset_name ,mask_fields, metric_fns =[metrics.bleu,metrics.rouge])


seqio.MixtureRegistry.add(
  "may6_train_round2",
  [('may6_train_round2_v1',1),
  ('may6_train_round2_qa',1),
  ('april25_turbo_annotations_v1',1)])


# ==================================== may8 iterative ======================================
# multi-round iterative training tasks
#
# Uses the 20k turbo annotations rather than the 100k (sampled from different base sets)
# ALSO: uses a smaller ratio for annotations


annotation_task = 'april11_model_annotation_fs_turbo'

experiment_name = 'may8'

## predefine tasks for many rounds so we don't have to keep hacking into this
for round_ in range(10):

    dataset_name = '{}_train_round{}_v1'.format(experiment_name, round_)
    file_template = 'gs://ai2-mosaic-private/peter-skd-2023/iterative_runs/may6/data/round0/train_dataset_v1_{}.tsv'
    input_files = {'train':file_template.format('train'),
                'test':file_template.format('test'),
                'validation':file_template.format('val')}
    mask_fields = ['context','inference']
    build_task(input_files, dataset_name ,mask_fields, metric_fns =[metrics.bleu,metrics.rouge])


    dataset_name = '{}_train_round{}_qa'.format(experiment_name,round_)
    file_template = 'gs://ai2-mosaic-private/peter-skd-2023/iterative_runs/may6/data/round0/train_dataset_qa_{}.tsv'
    input_files = {'train':file_template.format('train'),
                'test':file_template.format('test'),
                'validation':file_template.format('val')}
    mask_fields = ['context','query','inference']
    build_task(input_files, dataset_name ,mask_fields, metric_fns =[metrics.bleu,metrics.rouge])


    dataset_name = '{}_train_round{}_annotation'.format(experiment_name,round_)
    file_template = 'gs://ai2-mosaic-private/peter-skd-2023/iterative_runs/may6/data/round0/data_to_score.tsv'
    input_files = {'train':file_template,
                'test':file_template,
                'validation':file_template}
    mask_fields =  ['plausibility']
    build_task(input_files, dataset_name ,mask_fields, metric_fns =[metrics.bleu,metrics.rouge], tsv_fields=['context','query','inference','plausibility','split','generation_round','plausibility_p','index','label'])



    seqio.MixtureRegistry.add(
      '{}_train_round{}'.format(experiment_name, round_),
      [('{}_train_round{}_v1'.format(experiment_name, round_),2),
      ('{}_train_round{}_qa'.format(experiment_name,round_),2),
      (annotation_task,1)])


    
# ==================================== may15 no-none ======================================
# Uses the old dataset (no nones) + 20k human annotations


annotation_task = 'round2_critic_feb10'

experiment_name = 'may15_nonone_human20k'


## predefine tasks for many rounds so we don't have to keep hacking into this
for round_ in range(10):

    dataset_name = '{}_train_round{}_v1'.format(experiment_name, round_)
    file_template = 'gs://ai2-mosaic-private/peter-skd-2023/iterative_runs/{}/round{}/data/train_dataset_v1_'.format(experiment_name, round_) +'{}.tsv'
    input_files = {'train':file_template.format('train'),
                'test':file_template.format('test'),
                'validation':file_template.format('val')}
    mask_fields = ['context','inference']
    build_task(input_files, dataset_name ,mask_fields, metric_fns =[metrics.bleu,metrics.rouge])


    dataset_name = '{}_train_round{}_qa'.format(experiment_name,round_)
    file_template = 'gs://ai2-mosaic-private/peter-skd-2023/iterative_runs/{}/round{}/data/train_dataset_qa_'.format(experiment_name, round_) +'{}.tsv'
    input_files = {'train':file_template.format('train'),
                'test':file_template.format('test'),
                'validation':file_template.format('val')}
    mask_fields = ['context','query','inference']
    build_task(input_files, dataset_name ,mask_fields, metric_fns =[metrics.bleu,metrics.rouge])


    dataset_name = '{}_train_round{}_annotation'.format(experiment_name,round_)
    file_template = 'gs://ai2-mosaic-private/peter-skd-2023/iterative_runs/{}/data/round{}/data_to_score.tsv'.format(experiment_name, round_)
    input_files = {'train':file_template,
                'test':file_template,
                'validation':file_template}
    mask_fields =  ['plausibility']
    build_task(input_files, dataset_name ,mask_fields, metric_fns =[metrics.bleu,metrics.rouge], tsv_fields=['context','query','inference','plausibility','split','generation_round','plausibility_p','index','label'])



    ## NOTE!! IN THE NO-NONE, we only have QA-style data (no "none" queries)
    seqio.MixtureRegistry.add(
      '{}_train_round{}'.format(experiment_name, round_),
      [('{}_train_round{}_qa'.format(experiment_name,round_),2),
      (annotation_task,1)])
    
    
# ==================================== may15 nnone ======================================
# Uses the old dataset (no nones) + 20k human annotations


annotation_task = 'round2_critic_feb10'

experiment_name = 'may15_none_human20k'


## predefine tasks for many rounds so we don't have to keep hacking into this
for round_ in range(10):

    dataset_name = '{}_train_round{}_v1'.format(experiment_name, round_)
    file_template = 'gs://ai2-mosaic-private/peter-skd-2023/iterative_runs/{}/round{}/data/train_dataset_v1_'.format(experiment_name, round_) +'{}.tsv'
    input_files = {'train':file_template.format('train'),
                'test':file_template.format('test'),
                'validation':file_template.format('val')}
    mask_fields = ['context','inference']
    build_task(input_files, dataset_name ,mask_fields, metric_fns =[metrics.bleu,metrics.rouge])


    dataset_name = '{}_train_round{}_qa'.format(experiment_name,round_)
    file_template = 'gs://ai2-mosaic-private/peter-skd-2023/iterative_runs/{}/round{}/data/train_dataset_qa_'.format(experiment_name, round_) +'{}.tsv'
    input_files = {'train':file_template.format('train'),
                'test':file_template.format('test'),
                'validation':file_template.format('val')}
    mask_fields = ['context','query','inference']
    build_task(input_files, dataset_name ,mask_fields, metric_fns =[metrics.bleu,metrics.rouge])


    dataset_name = '{}_train_round{}_annotation'.format(experiment_name,round_)
    file_template = 'gs://ai2-mosaic-private/peter-skd-2023/iterative_runs/{}/data/round{}/data_to_score.tsv'.format(experiment_name, round_)
    input_files = {'train':file_template,
                'test':file_template,
                'validation':file_template}
    mask_fields =  ['plausibility']
    build_task(input_files, dataset_name ,mask_fields, metric_fns =[metrics.bleu,metrics.rouge], tsv_fields=['context','query','inference','plausibility','split','generation_round','plausibility_p','index','label'])



    seqio.MixtureRegistry.add(
      '{}_train_round{}'.format(experiment_name, round_),
      [('{}_train_round{}_v1'.format(experiment_name, round_),2),
      ('{}_train_round{}_qa'.format(experiment_name,round_),2),
      (annotation_task,1)])
    
    
# ==================================== may15 nnone gpt4 ======================================
# Uses the old dataset (no nones) + 20k human annotations


annotation_task = 'april7_model_annotation_gpt4'

experiment_name = 'may15_none_gpt4'


## predefine tasks for many rounds so we don't have to keep hacking into this
for round_ in range(10):

    dataset_name = '{}_train_round{}_v1'.format(experiment_name, round_)
    file_template = 'gs://ai2-mosaic-private/peter-skd-2023/iterative_runs/{}/round{}/data/train_dataset_v1_'.format(experiment_name, round_) +'{}.tsv'
    input_files = {'train':file_template.format('train'),
                'test':file_template.format('test'),
                'validation':file_template.format('val')}
    mask_fields = ['context','inference']
    build_task(input_files, dataset_name ,mask_fields, metric_fns =[metrics.bleu,metrics.rouge])


    dataset_name = '{}_train_round{}_qa'.format(experiment_name,round_)
    file_template = 'gs://ai2-mosaic-private/peter-skd-2023/iterative_runs/{}/round{}/data/train_dataset_qa_'.format(experiment_name, round_) +'{}.tsv'
    input_files = {'train':file_template.format('train'),
                'test':file_template.format('test'),
                'validation':file_template.format('val')}
    mask_fields = ['context','query','inference']
    build_task(input_files, dataset_name ,mask_fields, metric_fns =[metrics.bleu,metrics.rouge])


    dataset_name = '{}_train_round{}_annotation'.format(experiment_name,round_)
    file_template = 'gs://ai2-mosaic-private/peter-skd-2023/iterative_runs/{}/data/round{}/data_to_score.tsv'.format(experiment_name, round_)
    input_files = {'train':file_template,
                'test':file_template,
                'validation':file_template}
    mask_fields =  ['plausibility']
    build_task(input_files, dataset_name ,mask_fields, metric_fns =[metrics.bleu,metrics.rouge], tsv_fields=['context','query','inference','plausibility','split','generation_round','plausibility_p','index','label'])



    seqio.MixtureRegistry.add(
      '{}_train_round{}'.format(experiment_name, round_),
      [('{}_train_round{}_v1'.format(experiment_name, round_),2),
      ('{}_train_round{}_qa'.format(experiment_name,round_),2),
      (annotation_task,1)])
    
    
    


    
    
    
    
    
    
    
# ==================================== may16 no-none ======================================
# Uses the old dataset (no nones) + 20k human annotations


annotation_task = 'round2_critic_feb10'

experiment_name = 'may16_nonone_human20k'


## predefine tasks for many rounds so we don't have to keep hacking into this
for round_ in range(10):

    dataset_name = '{}_train_round{}_v1'.format(experiment_name, round_)
    file_template = 'gs://ai2-mosaic-private/peter-skd-2023/iterative_runs/{}/round{}/data/train_dataset_v1_'.format(experiment_name, round_) +'{}.tsv'
    input_files = {'train':file_template.format('train'),
                'test':file_template.format('test'),
                'validation':file_template.format('val')}
    mask_fields = ['context','inference']
    build_task(input_files, dataset_name ,mask_fields, metric_fns =[metrics.bleu,metrics.rouge])


    dataset_name = '{}_train_round{}_qa'.format(experiment_name,round_)
    file_template = 'gs://ai2-mosaic-private/peter-skd-2023/iterative_runs/{}/round{}/data/train_dataset_qa_'.format(experiment_name, round_) +'{}.tsv'
    input_files = {'train':file_template.format('train'),
                'test':file_template.format('test'),
                'validation':file_template.format('val')}
    mask_fields = ['context','query','inference']
    build_task(input_files, dataset_name ,mask_fields, metric_fns =[metrics.bleu,metrics.rouge])


    dataset_name = '{}_train_round{}_annotation'.format(experiment_name,round_)
    file_template = 'gs://ai2-mosaic-private/peter-skd-2023/iterative_runs/{}/data/round{}/data_to_score.tsv'.format(experiment_name, round_)
    input_files = {'train':file_template,
                'test':file_template,
                'validation':file_template}
    mask_fields =  ['plausibility']
    build_task(input_files, dataset_name ,mask_fields, metric_fns =[metrics.bleu,metrics.rouge], tsv_fields=['context','query','inference','plausibility','split','generation_round','plausibility_p','index','label'])



    ## NOTE!! IN THE NO-NONE, we only have QA-style data (no "none" queries)
    seqio.MixtureRegistry.add(
      '{}_train_round{}'.format(experiment_name, round_),
      [('{}_train_round{}_qa'.format(experiment_name,round_),6),
      (annotation_task,1)])
    
    
# ==================================== may16 nnone ======================================
# Uses the old dataset (no nones) + 20k human annotations


annotation_task = 'round2_critic_feb10'

experiment_name = 'may16_none_human20k'


## predefine tasks for many rounds so we don't have to keep hacking into this
for round_ in range(10):

    dataset_name = '{}_train_round{}_v1'.format(experiment_name, round_)
    file_template = 'gs://ai2-mosaic-private/peter-skd-2023/iterative_runs/{}/round{}/data/train_dataset_v1_'.format(experiment_name, round_) +'{}.tsv'
    input_files = {'train':file_template.format('train'),
                'test':file_template.format('test'),
                'validation':file_template.format('val')}
    mask_fields = ['context','inference']
    build_task(input_files, dataset_name ,mask_fields, metric_fns =[metrics.bleu,metrics.rouge])


    dataset_name = '{}_train_round{}_qa'.format(experiment_name,round_)
    file_template = 'gs://ai2-mosaic-private/peter-skd-2023/iterative_runs/{}/round{}/data/train_dataset_qa_'.format(experiment_name, round_) +'{}.tsv'
    input_files = {'train':file_template.format('train'),
                'test':file_template.format('test'),
                'validation':file_template.format('val')}
    mask_fields = ['context','query','inference']
    build_task(input_files, dataset_name ,mask_fields, metric_fns =[metrics.bleu,metrics.rouge])


    dataset_name = '{}_train_round{}_annotation'.format(experiment_name,round_)
    file_template = 'gs://ai2-mosaic-private/peter-skd-2023/iterative_runs/{}/data/round{}/data_to_score.tsv'.format(experiment_name, round_)
    input_files = {'train':file_template,
                'test':file_template,
                'validation':file_template}
    mask_fields =  ['plausibility']
    build_task(input_files, dataset_name ,mask_fields, metric_fns =[metrics.bleu,metrics.rouge], tsv_fields=['context','query','inference','plausibility','split','generation_round','plausibility_p','index','label'])



    seqio.MixtureRegistry.add(
      '{}_train_round{}'.format(experiment_name, round_),
      [('{}_train_round{}_v1'.format(experiment_name, round_),2),
      ('{}_train_round{}_qa'.format(experiment_name,round_),4),
      (annotation_task,1)])

