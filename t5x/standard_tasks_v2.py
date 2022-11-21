from atomic_seqio_utils import key_prefixes, fill_missing_fields, mask_each_field, combine_inputs

from t5.data import preprocessors
import t5
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
                       "plausibility":32096,
                        "relevance":32095}


vocabulary = t5.data.get_default_vocabulary()
fields = list(fields_to_sentinels.keys())


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


def build_task(input_files, task_name, tsv_fields, mask_fields=None, p_full=0.5):
    

    # if no option given, allow masking of all fields
    if mask_fields is None:
        mask_fields = nonempty_fields
        
        
    # get field mask options, i.e. any mask bitmap where all masks are on "mask_fields"
    # i.e. don't mask any fields not in mask_fields
    # also, require at least one mask
    mask_field_inds = [fields.index(field) for field in mask_fields]
    field_mask_options =  [list(l) for l in itertools.product([0, 1], repeat=5) if (sum([l[ind] for ind in mask_field_inds]) == sum(l)) and sum(l) > 0]

    
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
        seqio.preprocessors.tokenize,
        
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

    ],
    output_features=DEFAULT_OUTPUT_FEATURES)
    
    
    
    
    
    
    
    
    




datasets = []
    
'''
Datasets -- ATOMIC-10X labeled
'''

dataset_name = 'ATOMIX10X_labeled'
input_files = {'train':'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/standard_format_data/atomic10x_v0/train_labelled.tsv',
            'test':'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/standard_format_data/atomic10x_v0/test_labelled.tsv',
            'validation':'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/standard_format_data/atomic10x_v0/val_labelled.tsv'}
tsv_fields = ['premise', 'hypothesis','plausibility','question','relevance']

nonempty_fields = ['premise','hypothesis','plausibility','question']

share = 1

datasets.append({'input_files': input_files, 'dataset_name':dataset_name,
                'tsv_fields':tsv_fields,
                'nonempty_fields':nonempty_fields,
                'share':share})





'''
Datasets -- ATOMIC-10X unlabeled
'''

dataset_name = 'ATOMIX10X_unlabeled'
input_files = {'train':'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/standard_format_data/atomic10x_v0/train_unlabelled.tsv',
            'test':'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/standard_format_data/atomic10x_v0/test_unlabelled.tsv',
            'validation':'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/standard_format_data/atomic10x_v0/val_unlabelled.tsv'}

tsv_fields = ['premise', 'hypothesis','question','relevance']

nonempty_fields = ['premise','hypothesis','question']

share = 8

datasets.append({'input_files': input_files, 'dataset_name':dataset_name,
                'tsv_fields':tsv_fields,
                'nonempty_fields':nonempty_fields,
                'share':share})



'''
Datasets -- ATOMIC2020 
'''

dataset_name = 'ATOMIX2020_unlabeled'
input_files = {'train':'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/standard_format_data/atomic2020_v0/train.tsv',
            'test':'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/standard_format_data/atomic2020_v0/test.tsv',
            'validation':'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/standard_format_data/atomic2020_v0/val.tsv'}

tsv_fields = ['premise', 'hypothesis','question','relevance']

nonempty_fields = ['premise','hypothesis','question','relevance']

share = 8

datasets.append({'input_files': input_files, 'dataset_name':dataset_name,
                'tsv_fields':tsv_fields,
                'nonempty_fields':nonempty_fields,
                'share':share})


'''
generated_round0
'''

dataset_name = 'generated_round0'
input_files = {'train':'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/standard_format_data/generated/train.tsv',
            'test':'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/standard_format_data/generated/test.tsv',
            'validation':'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/standard_format_data/generated/val.tsv'}

tsv_fields = ['premise', 'hypothesis']

nonempty_fields = ['premise', 'hypothesis']

share = 2

datasets.append({'input_files': input_files, 'dataset_name':dataset_name,
                'tsv_fields':tsv_fields,
                'nonempty_fields':nonempty_fields,
                'share':share})




'''
generated_round1 premise + hypo
'''

dataset_name = 'generated_round1_prem_hypo'
input_files = {'train':'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/standard_format_data/generated_v1/train_v1-premises_final.tsv',
            'test':'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/standard_format_data/generated_v1/test_v1-premises_final.tsv',
            'validation':'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/standard_format_data/generated_v1/val_v1-premises_final.tsv'}

tsv_fields = ['premise', 'hypothesis']

nonempty_fields = ['premise', 'hypothesis']

share = 4

datasets.append({'input_files': input_files, 'dataset_name':dataset_name,
                'tsv_fields':tsv_fields,
                'nonempty_fields':nonempty_fields,
                'share':share})

'''
generated_round1 prem + question + hypo
'''

dataset_name = 'generated_round1_prem_qa_hypo'
input_files = {'train':'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/standard_format_data/generated_v1/train_qa-premises_final.tsv',
            'test':'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/standard_format_data/generated_v1/test_qa-premises_final.tsv',
            'validation':'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/standard_format_data/generated_v1/val_qa-premises_final.tsv'}

tsv_fields = ['premise','question', 'hypothesis']

nonempty_fields = ['premise','question', 'hypothesis']

share = 4

datasets.append({'input_files': input_files, 'dataset_name':dataset_name,
                'tsv_fields':tsv_fields,
                'nonempty_fields':nonempty_fields,
                'share':share})





'''
First, generate training tasks
'''
for dataset in datasets:
    task_name = 'train_{}'.format(dataset['dataset_name'])
    build_task(dataset['input_files'], task_name,dataset['tsv_fields'],dataset['nonempty_fields'])
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
  "standard_v2_train_mix",
  train_tasks)

seqio.MixtureRegistry.add(
  "standard_v2_eval_mix",
  train_tasks + eval_tasks)
    
