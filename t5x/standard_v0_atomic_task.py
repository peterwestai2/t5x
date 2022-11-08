from t5.data import preprocessors
import t5
import seqio
import functools
import tensorflow as tf
import random
import itertools

'''

this is just using atomic 10x and atomic 2020 data with templates

using a standard format with premise, quesiton, hypothesis, plausibility, and relevance

'''

'''

ATOMIC 2020 version

'''
split_map = {'train':'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/standard_format_data/atomic2020_v0/train.tsv',
            'test':'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/standard_format_data/atomic2020_v0/test.tsv',
            'validation':'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/standard_format_data/atomic2020_v0/val.tsv'}

vocabulary = t5.data.get_default_vocabulary()


DEFAULT_OUTPUT_FEATURES = {
    "inputs":
        seqio.Feature(
            vocabulary=vocabulary, add_eos=True),
    "targets":
        seqio.Feature(
            vocabulary=vocabulary, add_eos=True),
    
}

fields_to_sentinels = {"premise":32099,
                       "question":32098,
                       "hypothesis":32097,
                       "plausibility":32096,
                        "relevance":32095}

fields = list(fields_to_sentinels.keys())
print('fields: {}'.format(fields))

tsv_fields = ['premise', 'hypothesis','question','relevance']

'''

Define the different versions (by field) of masking that are allowed

we will sample from these uniformly

Each of these corresponds to which of the n fields are masked
e.g. [0,1,0] would mean only the second field is masked (i.e. set to 1)

'''
# using itertools to enumerate the options for this...
field_mask_options =  [list(l) for l in itertools.product([0, 1], repeat=5) if l[3] == 0]









'''

define output features

'''


# for each field, add an output feature
for field in fields:
    DEFAULT_OUTPUT_FEATURES.update({'{}'.format(field):
                                    seqio.Feature( 
                                        vocabulary=vocabulary, add_eos=False),})
    DEFAULT_OUTPUT_FEATURES.update({'{}_prefix'.format(field):
                                    seqio.Feature( 
                                        vocabulary=vocabulary, add_eos=False),})


'''

Helper functions for constructing the dataset

'''

from atomic_seqio_utils import key_prefixes, fill_missing_fields, mask_each_field, combine_inputs

seqio.TaskRegistry.add(
    "standard_v0_atomic_2020",
    seqio.TextLineDataSource(split_map,skip_header_lines=1,),
    preprocessors=[
        
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
                          field_mask_options = field_mask_options, fields=fields,fields_to_sentinels=fields_to_sentinels), 
        
        # combine field-specific input/targets into general ones
        functools.partial(combine_inputs, fields=fields),
        
        # add eos token
        seqio.preprocessors.append_eos,
        
        # remove some extraneous features
        functools.partial(
            preprocessors.rekey, key_map={key:key for key in DEFAULT_OUTPUT_FEATURES.keys()}),

    ],
    output_features=DEFAULT_OUTPUT_FEATURES)






'''

ATOMIC 10x unlabelled version

'''
split_map = {'train':'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/standard_format_data/atomic10x_v0/train_unlabelled.tsv',
            'test':'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/standard_format_data/atomic10x_v0/test_unlabelled.tsv',
            'validation':'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/standard_format_data/atomic10x_v0/val_unlabelled.tsv'}

vocabulary = t5.data.get_default_vocabulary()


DEFAULT_OUTPUT_FEATURES = {
    "inputs":
        seqio.Feature(
            vocabulary=vocabulary, add_eos=True),
    "targets":
        seqio.Feature(
            vocabulary=vocabulary, add_eos=True),
    
}

fields_to_sentinels = {"premise":32099,
                       "question":32098,
                       "hypothesis":32097,
                       "plausibility":32096,
                        "relevance":32095}

fields = list(fields_to_sentinels.keys())
print('fields: {}'.format(fields))

tsv_fields = ['premise', 'hypothesis','question','relevance']

'''

Define the different versions (by field) of masking that are allowed

we will sample from these uniformly

Each of these corresponds to which of the n fields are masked
e.g. [0,1,0] would mean only the second field is masked (i.e. set to 1)

'''
field_mask_options =  [list(l) for l in itertools.product([0, 1], repeat=5) if (l[3] == 0 and l[4]==0)]








'''

define output features

'''


# for each field, add an output feature
for field in fields:
    DEFAULT_OUTPUT_FEATURES.update({'{}'.format(field):
                                    seqio.Feature( 
                                        vocabulary=vocabulary, add_eos=False),})
    DEFAULT_OUTPUT_FEATURES.update({'{}_prefix'.format(field):
                                    seqio.Feature( 
                                        vocabulary=vocabulary, add_eos=False),})


'''

Helper functions for constructing the dataset

'''

from atomic_seqio_utils import key_prefixes, fill_missing_fields, mask_each_field, combine_inputs

seqio.TaskRegistry.add(
    "standard_v0_atomic_10X_unlabelled",
    seqio.TextLineDataSource(split_map,skip_header_lines=1,),
    preprocessors=[
        
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
                          field_mask_options = field_mask_options, fields=fields,fields_to_sentinels=fields_to_sentinels), 
        
        # combine field-specific input/targets into general ones
        functools.partial(combine_inputs, fields=fields),
        
        # add eos token
        seqio.preprocessors.append_eos,
        
        # remove some extraneous features
        functools.partial(
            preprocessors.rekey, key_map={key:key for key in DEFAULT_OUTPUT_FEATURES.keys()}),

    ],
    output_features=DEFAULT_OUTPUT_FEATURES)





'''

ATOMIC 10x labelled version

'''
split_map = {'train':'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/standard_format_data/atomic10x_v0/train_labelled.tsv',
            'test':'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/standard_format_data/atomic10x_v0/test_labelled.tsv',
            'validation':'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/standard_format_data/atomic10x_v0/val_labelled.tsv'}

vocabulary = t5.data.get_default_vocabulary()


DEFAULT_OUTPUT_FEATURES = {
    "inputs":
        seqio.Feature(
            vocabulary=vocabulary, add_eos=True),
    "targets":
        seqio.Feature(
            vocabulary=vocabulary, add_eos=True),
    
}

fields_to_sentinels = {"premise":32099,
                       "question":32098,
                       "hypothesis":32097,
                       "plausibility":32096,
                        "relevance":32095}

fields = list(fields_to_sentinels.keys())
print('fields: {}'.format(fields))

tsv_fields = ['premise', 'hypothesis','plausibility','question','relevance']

'''

Define the different versions (by field) of masking that are allowed

we will sample from these uniformly

Each of these corresponds to which of the n fields are masked
e.g. [0,1,0] would mean only the second field is masked (i.e. set to 1)

'''
field_mask_options =  [list(l) for l in itertools.product([0, 1], repeat=5) if (l[4]==0)]








'''

define output features

'''


# for each field, add an output feature
for field in fields:
    DEFAULT_OUTPUT_FEATURES.update({'{}'.format(field):
                                    seqio.Feature( 
                                        vocabulary=vocabulary, add_eos=False),})
    DEFAULT_OUTPUT_FEATURES.update({'{}_prefix'.format(field):
                                    seqio.Feature( 
                                        vocabulary=vocabulary, add_eos=False),})


'''

Helper functions for constructing the dataset

'''

from atomic_seqio_utils import key_prefixes, fill_missing_fields, mask_each_field, combine_inputs

seqio.TaskRegistry.add(
    "standard_v0_atomic_10X_labelled",
    seqio.TextLineDataSource(split_map,skip_header_lines=1,),
    preprocessors=[
        
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
                          field_mask_options = field_mask_options, fields=fields,fields_to_sentinels=fields_to_sentinels), 
        
        # combine field-specific input/targets into general ones
        functools.partial(combine_inputs, fields=fields),
        
        # add eos token
        seqio.preprocessors.append_eos,
        
        # remove some extraneous features
        functools.partial(
            preprocessors.rekey, key_map={key:key for key in DEFAULT_OUTPUT_FEATURES.keys()}),

    ],
    output_features=DEFAULT_OUTPUT_FEATURES)






'''

generated version

'''
split_map = {'train':'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/standard_format_data/generated/train.tsv',
            'test':'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/standard_format_data/generated/test.tsv',
            'validation':'gs://ai2-mosaic-public/projects/symbolic-knowledge-decoding/standard_format_data/generated/val.tsv'}

vocabulary = t5.data.get_default_vocabulary()


DEFAULT_OUTPUT_FEATURES = {
    "inputs":
        seqio.Feature(
            vocabulary=vocabulary, add_eos=True),
    "targets":
        seqio.Feature(
            vocabulary=vocabulary, add_eos=True),
    
}

fields_to_sentinels = {"premise":32099,
                       "question":32098,
                       "hypothesis":32097,
                       "plausibility":32096,
                        "relevance":32095}

fields = list(fields_to_sentinels.keys())
print('fields: {}'.format(fields))

tsv_fields = ['premise', 'hypothesis']

'''

Define the different versions (by field) of masking that are allowed

we will sample from these uniformly

Each of these corresponds to which of the n fields are masked
e.g. [0,1,0] would mean only the second field is masked (i.e. set to 1)

'''
# using itertools to enumerate the options for this...
field_mask_options =  [list(l) for l in itertools.product([0, 1], repeat=5) if (l[1] == 0) and (l[3] == 0) and (l[4] == 0)]









'''

define output features

'''


# for each field, add an output feature
for field in fields:
    DEFAULT_OUTPUT_FEATURES.update({'{}'.format(field):
                                    seqio.Feature( 
                                        vocabulary=vocabulary, add_eos=False),})
    DEFAULT_OUTPUT_FEATURES.update({'{}_prefix'.format(field):
                                    seqio.Feature( 
                                        vocabulary=vocabulary, add_eos=False),})


'''

Helper functions for constructing the dataset

'''

from atomic_seqio_utils import key_prefixes, fill_missing_fields, mask_each_field, combine_inputs

seqio.TaskRegistry.add(
    "standard_v0_generated_0",
    seqio.TextLineDataSource(split_map,skip_header_lines=1,),
    preprocessors=[
        
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
                          field_mask_options = field_mask_options, fields=fields,fields_to_sentinels=fields_to_sentinels), 
        
        # combine field-specific input/targets into general ones
        functools.partial(combine_inputs, fields=fields),
        
        # add eos token
        seqio.preprocessors.append_eos,
        
        # remove some extraneous features
        functools.partial(
            preprocessors.rekey, key_map={key:key for key in DEFAULT_OUTPUT_FEATURES.keys()}),

    ],
    output_features=DEFAULT_OUTPUT_FEATURES)






seqio.MixtureRegistry.add(
  "standard_v0_atomic_mix",
  [("standard_v0_atomic_10X_labelled", 1), ("standard_v0_atomic_2020", 8),("standard_v0_atomic_10X_unlabelled", 8),("standard_v0_generated_0",4)]
)




