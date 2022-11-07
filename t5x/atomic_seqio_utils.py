from t5.data import preprocessors
import t5
import seqio
import functools
import tensorflow as tf
import random
import gin


'''

This file contains functions needed to convert into 
standard commonsense format for seqio

'''


def generate_token_mask_skd_v0(tokens, do_mask, p_full = 0.5):
    '''
    
    the idea behind this helper function is to randomly mask the given field (tokens)
    
    p_full is the probability of generating a mask of the full field
    
    with probability p_full we mask the full field
    with probability 1-p_full we mask between 1 and the full length 
    (so p_full is a slight underestimate of the chance of full masking. this
    makes it simpler in the case where, e.g. the field only has 1 token)
    
    The output of this function is a bitmap tensor of the same shape as 'tokens',
    with a contiguous span of 1's indicating a masked region
    
    no_mask allows us to turn off masking in this case. To be used for unmasked fields
    
    '''
    
    
    tokens_length = tf.size(tokens)
    possible_lengths = tf.range(tf.size(tokens)) + 1

    # possible lengths, if p_full roll is false
    not_full_length = tf.random.shuffle(possible_lengths)[0]

    # length, if p_full roll is true (i.e. full length)
    full_length = possible_lengths[-1]

    roll_p_full = tf.random.uniform([1])
    roll_result = tf.cast(roll_p_full < p_full, possible_lengths.dtype)[0]

    mask_length =  roll_result*full_length  + (1-roll_result)*not_full_length

    
    # if no_mask, set the mask length to 0 (i.e. mask nothing)
    #mask_length =  (1-do_mask)*full_length  
    
    # need the max in there in case we set length to 0 (i.e. empty)
    possible_span_start = tf.range(tf.size(tokens) - mask_length + 1)
    

    mask_start_ind = tf.random.shuffle(possible_span_start)[0]

    # Get a mask of length mask_length, starting at mask_start_ind
    mask = tf.math.logical_and((tf.range(tf.size(tokens)) >= mask_start_ind),  (tf.range(tf.size(tokens)) < mask_start_ind + mask_length))

    # if not masking, make the mask all false 
    #anti_mask = (tf.ones_like(mask)*(no_mask) == 0)
    anti_mask = (tf.ones_like(mask,dtype = tokens.dtype)*(do_mask)) == 1
    return tf.logical_and(anti_mask, mask), anti_mask




'''

Below are functions for converting masked fields into input/target

'''

@gin.configurable()
def noise_span_to_sentinel_single(tokens, noise_mask, sentinel):
    prev_token_is_noise = tf.pad(noise_mask[:-1], [[1, 0]])

    first_noise_tokens = tf.logical_and(
        noise_mask, tf.logical_not(prev_token_is_noise))
    subsequent_noise_tokens = tf.logical_and(noise_mask, prev_token_is_noise)


    tokens = tf.where(first_noise_tokens, sentinel, tokens)
    return tf.boolean_mask(tokens, tf.logical_not(subsequent_noise_tokens))

@gin.configurable()
def nonnoise_span_to_sentinel_single(tokens, noise_mask, sentinel):
    return tf.concat([[sentinel],tokens[noise_mask]], axis=-1)




'''

seqio preprocessing functions for atomic data

'''


def key_prefixes(ds, fields):
    def key_prefixes_(ex):

        for field in fields:
            ## add the prefix for each field. Each field begins with eid 99
            ex.update({'{}_prefix'.format(field): tf.strings.join([' <extra_id_99> {}: '.format(field)])})
        return ex
    return ds.map(key_prefixes_, num_parallel_calls=tf.data.experimental.AUTOTUNE)

def fill_missing_fields(ds, fields):
    def fill_missing_fields_(ex):

        for field in fields + ['inputs','targets']:
            if field not in ex.keys():
                ex[field] = ' none'
        return ex
    return ds.map(fill_missing_fields_, num_parallel_calls=tf.data.experimental.AUTOTUNE)

def mask_each_field(ds, field_mask_options, fields,fields_to_sentinels):
    
    def mask_each_field_(ex):
        field_mask_options_ = tf.constant(field_mask_options, dtype=ex['inputs'].dtype)
        field_mask_options_ind = tf.random.shuffle(tf.range(field_mask_options_.shape[0]))[0]
        field_mask_option = field_mask_options_[field_mask_options_ind]
        
        ex['field_mask'] = field_mask_option
        
        for i, field in enumerate(fields):
            ex['{}_mask_option'.format(field)] = field_mask_option[i]
            mask, anti_mask = generate_token_mask_skd_v0(ex[field],field_mask_option[i])
            ex['{}_mask'.format(field)] = mask
            ex['{}_antimask'.format(field)] = anti_mask
            ex['{}_inputs'.format(field)] = noise_span_to_sentinel_single(ex[field], mask, fields_to_sentinels[field])
            ex['{}_targets'.format(field)] = nonnoise_span_to_sentinel_single(ex[field], mask, fields_to_sentinels[field])
        
        return ex
        
    return ds.map(mask_each_field_, num_parallel_calls=tf.data.experimental.AUTOTUNE)     

 
        
def combine_inputs(ds, fields):
    def combine_inputs_(ex):


        input_list = []
        target_list = []

        for field in fields:
            input_list.append(ex['{}_prefix'.format(field)])
            input_list.append(ex['{}_inputs'.format(field)])

            target_list.append(ex['{}_targets'.format(field)])


        ex['inputs'] = tf.concat(input_list,axis=-1)
        ex['targets'] = tf.concat(target_list,axis=-1)

        return ex
    return ds.map(combine_inputs_, num_parallel_calls=tf.data.experimental.AUTOTUNE)   
