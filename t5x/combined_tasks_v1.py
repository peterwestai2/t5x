'''

Backup of the updates I made to the new standard tasks, while I try to fix whatever broke it

'''


from atomic_seqio_utils import key_prefixes, fill_missing_fields, mask_each_field, combine_inputs

from standard_utils_v1 import build_task

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




# ==================================== critic sample ablation ======================================
# Try different numbers of annotated points for training the critic model
#
# This makes:
#   critic_ablation_500
#   critic_ablation_1k
#   critic_ablation_4k
#   critic_ablation_8k
#
# These all have the same dev/test set but different sizes of training data
# The idea is to figure out how much adding human annotation will help


for scale in ['500','1k','4k','8k']:
  

  dataset_name = 'critic_ablation_{}'.format(scale)


  file_template = 'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/jan_9_2022_dataset/human_annotated_multival_{}.tsv'
  input_files = {'train':file_template.format('train'),
              'test':file_template.format('test'),
              'validation':file_template.format('val')}
  mask_fields =  ['reasonable'] 
  
  build_task(input_files, dataset_name ,mask_fields, metric_fns =[metrics.bleu,metrics.rouge])



    
