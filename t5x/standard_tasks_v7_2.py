'''

Backup of the updates I made to the new standard tasks, while I try to fix whatever broke it

'''


from atomic_seqio_utils import key_prefixes, fill_missing_fields, mask_each_field, combine_inputs

from t5.data import preprocessors
import t5
from t5.evaluation import metrics
import seqio
import functools
import tensorflow as tf
import random
import itertools

'''


Here, we define all tasks we will use for training and evaluation for standard v1
which adds evaluation-specific tasks and an evaluation mixture


'''




'''
First, Define general features etc.
'''

train_tasks = []
eval_tasks = []



## conversion of fields to per-field mask tokens
fields_to_sentinels = {"premise":32099,
                       "question":32098,
                       "hypothesis":32097,
                       "reasonable":32096}


vocabulary = t5.data.get_default_vocabulary()
fields = list(fields_to_sentinels.keys())


DEFAULT_OUTPUT_FEATURES_FINAL = {
    "inputs":
        seqio.Feature(
            vocabulary=vocabulary, add_eos=True),
    "targets":
        seqio.Feature(
            vocabulary=vocabulary, add_eos=True),
    
}

DEFAULT_OUTPUT_FEATURES = {
    "inputs":
        seqio.Feature(
            vocabulary=vocabulary, add_eos=True),
    "targets":
        seqio.Feature(
            vocabulary=vocabulary, add_eos=True),
    
}
# for each field, add an output feature
for field in fields:
    DEFAULT_OUTPUT_FEATURES.update({'{}'.format(field):
                                    seqio.Feature( 
                                        vocabulary=vocabulary, add_eos=False),})
    DEFAULT_OUTPUT_FEATURES.update({'{}_prefix'.format(field):
                                    seqio.Feature( 
                                        vocabulary=vocabulary, add_eos=False),})


    
    
def my_tokenize(dataset: tf.data.Dataset) -> tf.data.Dataset:
  """Encode output features with specified vocabularies.
  Passes through other features unchanged. Optionally passes through copy
  of original features with "_pretokenized" suffix added to the key.
  When `with_eos` is True and input features are ranked > 1, then an EOS is
  appended only to the last item of each 1-D sequence.
  Args:
    dataset: a tf.data.Dataset of examples to tokenize.
    output_features: a dict of Feature objects; their vocabulary attribute will
      be used to tokenize the specified features.
    copy_pretokenized: bool, whether to pass through copies of original features
      with "_pretokenized" suffix added to the key.
    with_eos: bool, whether to append EOS to the end of the sequence.
  Returns:
    a tf.data.Dataset
  """
  tokenize_fn = functools.partial(
      seqio.preprocessors.tokenize_impl,
      output_features=DEFAULT_OUTPUT_FEATURES,
      copy_pretokenized=False,
      with_eos=False)
  return seqio.utils.map_over_dataset(fn=tokenize_fn)(dataset)
    
def build_task(input_files, task_name, tsv_fields, mask_fields=None, p_full=0.5, field_mask_options = None, metric_fns = []):
    

    # if no option given, allow masking of all fields
    if mask_fields is None:
        mask_fields = nonempty_fields
        
        
    # get field mask options, i.e. any mask bitmap where all masks are on "mask_fields"
    # i.e. don't mask any fields not in mask_fields
    # also, require at least one mask
    mask_field_inds = [fields.index(field) for field in mask_fields]
    
    if field_mask_options is None:
        field_mask_options =  [list(l) for l in itertools.product([0, 1], repeat=len(fields)) if (sum([l[ind] for ind in mask_field_inds]) == sum(l)) and sum(l) > 0]

    

    
    # define the task
    seqio.TaskRegistry.add(
    task_name,
    seqio.TextLineDataSource(input_files,skip_header_lines=1,),
    preprocessors=[
        # parse input files
        functools.partial(
          t5.data.preprocessors.parse_tsv,
          field_names=tsv_fields),

        # preprocessing functions
        functools.partial(fill_missing_fields, fields=fields),
        functools.partial(key_prefixes, fields=fields),
    
        # tokenize 
        my_tokenize,

        
        # use random masking to generate inputs/targets for each field (similar to T5 objective)
        functools.partial(mask_each_field, 
                          field_mask_options = field_mask_options, fields=fields,fields_to_sentinels=fields_to_sentinels, p_full=p_full), 
        
        # combine field-specific input/targets into general ones
        functools.partial(combine_inputs, fields=fields),
        
        # add eos token
        seqio.preprocessors.append_eos,
        
        # remove some extraneous features
        functools.partial(
            preprocessors.rekey, key_map={key:key for key in DEFAULT_OUTPUT_FEATURES.keys()}),
        # remove some extraneous features
        #functools.partial(
        #    preprocessors.rekey, key_map={key:key for key in ['inputs','targets']}),

    ],
    metric_fns=metric_fns,
    #output_features=DEFAULT_OUTPUT_FEATURES)
    output_features=DEFAULT_OUTPUT_FEATURES_FINAL)
    
    
    
    
    
    
    
    
    




datasets = []
    
  
tsv_fields = ['premise','question','hypothesis','reasonable'] 
 
'''
Datasets -- ATOMIC-10X
'''

dataset_name = 'ATOMIC10X'


file_template = 'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/jan_9_2022_dataset/unannotated_ATOMIC10X_{}.tsv'
input_files = {'train':file_template.format('train'),
            'test':file_template.format('test'),
            'validation':file_template.format('val')}
nonempty_fields = ['premise','hypothesis','question']
share = 1
datasets.append({'input_files': input_files, 'dataset_name':dataset_name,
                'tsv_fields':tsv_fields,
                'nonempty_fields':nonempty_fields,
                'share':share})


'''
Datasets -- ATOMIC2020 
'''

dataset_name = 'ATOMIC2020'

file_template = 'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/jan_9_2022_dataset/unannotated_ATOMIC2020_{}.tsv'
input_files = {'train':file_template.format('train'),
            'test':file_template.format('test'),
            'validation':file_template.format('val')}
nonempty_fields = ['premise','hypothesis','question']
share = 1
datasets.append({'input_files': input_files, 'dataset_name':dataset_name,
                'tsv_fields':tsv_fields,
                'nonempty_fields':nonempty_fields,
                'share':share})


'''
Datasets -- generated 
'''

dataset_name = 'generated'

file_template = 'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/jan_9_2022_dataset/unannotated_generated2023_{}.tsv'
input_files = {'train':file_template.format('train'),
            'test':file_template.format('test'),
            'validation':file_template.format('val')}
nonempty_fields = ['premise','hypothesis','question']
share = 4
datasets.append({'input_files': input_files, 'dataset_name':dataset_name,
                'tsv_fields':tsv_fields,
                'nonempty_fields':nonempty_fields,
                'share':share})







'''
Datasets -- human annotated

The annotated dataset, but also masking every annotation field and never
masking the generative fields.

'''

dataset_name = 'human_annotated'


file_template = 'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/jan_9_2022_dataset/human_annotated_multival_{}.tsv'
input_files = {'train':file_template.format('train'),
            'test':file_template.format('test'),
            'validation':file_template.format('val')}
nonempty_fields =  ['premise','hypothesis','question','reasonable'] 
share = 2
datasets.append({'input_files': input_files, 'dataset_name':dataset_name,
                'tsv_fields':tsv_fields,
                'nonempty_fields':nonempty_fields,
                'share':share})



'''
Datasets -- human annotated critic

The annotated dataset, but also masking every annotation field and never
masking the generative fields. This multival version takes on values -1,0,1,2 instead of just 0,1

'''

dataset_name = 'human_annotated_multival_critic'


file_template = 'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/jan_9_2022_dataset/human_annotated_multival_{}.tsv'
input_files = {'train':file_template.format('train'),
            'test':file_template.format('test'),
            'validation':file_template.format('val')}
nonempty_fields =  ['premise','hypothesis','question','reasonable'] 
share = 6

dataset = {'input_files': input_files, 'dataset_name':dataset_name,
                'tsv_fields':tsv_fields,
                'nonempty_fields':nonempty_fields,
                'share':share}

task_name = 'train_{}'.format(dataset['dataset_name'])
build_task(dataset['input_files'], task_name,dataset['tsv_fields'],dataset['nonempty_fields'], field_mask_options=[[0,0,0,1]],
           metric_fns =[metrics.bleu,metrics.rouge])
train_tasks.append( (task_name, dataset['share']) )


'''
First, generate training tasks
'''
for dataset in datasets:
    task_name = 'train_{}'.format(dataset['dataset_name'])
    build_task(dataset['input_files'], task_name,dataset['tsv_fields'],dataset['nonempty_fields'], metric_fns =[metrics.bleu,metrics.rouge])
    train_tasks.append( (task_name, dataset['share']) )
    
    
'''
Next, generate eval tasks
'''
for dataset in datasets:
    nonempty_fields = dataset['nonempty_fields']
    for field in nonempty_fields:
        task_name = 'eval_{}_{}'.format(dataset['dataset_name'],field)
        build_task(dataset['input_files'], task_name,dataset['tsv_fields'],[field], p_full=1.0)
        eval_tasks.append( (task_name, 1) )
        
        
seqio.MixtureRegistry.add(
  "standard_v7_2_train_mix",
  train_tasks)

seqio.MixtureRegistry.add(
  "standard_v7_2_eval_mix",
  train_tasks + eval_tasks)
    
