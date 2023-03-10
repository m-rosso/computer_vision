###########################################################################################################
###########################################################################################################
#############################################################LIBRARIES##############################################################

import pandas as pd
import numpy as np
from datetime import datetime
import time
from itertools import product
import os

from keras.utils.layer_utils import count_params

###########################################################################################################
###########################################################################################################
#############################################################FUNCTIONS##############################################################

###########################################################################################################
# Function that calculates frequencies of elements in a list:

def frequency_list(_list):
    """
    Function that calculates frequencies of elements in a list:
    
    :param _list: .
    :type _list: list.
    
    :return: dictionary whose keys are the elements in a list and values are their frequencies.
    :rtype: dictionary.
    """
    _set = set(_list)
    freq_dict = {}

    # Loop over unique elements:
    for f in _set:
        freq_dict[f] = 0

        # Counting frequency:
        for i in _list:
            if i == f:
                freq_dict[f] += 1
    
    return freq_dict

###########################################################################################################
# Function that creates a list with different combinations of hyper-parameters:

def create_grid(params: dict, random_search: bool, n_samples: int = 10) -> list:
    """
    Creates a list with dictionaries whose keys are hyper-parameters and whose values are predefined during
    initialization.
    
    :param params: dictionary whose keys are hyper-parameters and values are lists with values for testing.
    :type params: dictionary.
    
    :param random_search: defines whether random search should be executed instead of grid search.
    :type random_search: boolean.
    
    :param n_samples: number of samples for random search.
    :type n_samples: integer (greater than zero).
    
    :return: list with combinations of values for hyper-parameters.
    :rtype: list.
    """
    # Grid search:
    if not random_search:
        grid_param = []
        
        # Loop over combinations of hyper-parameters:
        for i in eval('product(' + ','.join([str(params[p]) for p in params]) + ')'):
            grid_param.append(dict(zip(params.keys(), list(i))))

    # Random search:
    else:
        grid_param = []

        for i in range(1, n_samples+1):
            list_param = []

            for k in params.keys():
                try:
                    list_param.append(params[k].rvs(1)[0])
                except:
                    list_param.append(np.random.choice(params[k]))
            grid_param.append(dict(zip(params.keys(), list_param)))

    return grid_param

###########################################################################################################
# Function that returns the amount of time for running a block of code:

def running_time(start_time, end_time, print_time=True):
    """
    Function that returns the amount of time for running a block of code.
    
    :param start_time: time point when the code was initialized.
    :type start_time: datetime object obtained by executing "datetime.now()".

    :param end_time: time point when the code stopped its execution.
    :type end_time: datetime object obtained by executing "datetime.now()".

    :param print_unit: unit of time for presenting the runnning time.
    :type print_unit: string.
    
    :return: prints start, end time and running times, besides of returning running time in seconds.
    :rtype: integer.
    """
    if print_time:
        print('------------------------------------')
        print('\033[1mRunning time:\033[0m ' + str(round(((end_time - start_time).total_seconds())/60, 2)) +
              ' minutes.')
        print('Start time: ' + start_time.strftime('%Y-%m-%d') + ', ' + start_time.strftime('%H:%M:%S'))
        print('End time: ' + end_time.strftime('%Y-%m-%d') + ', ' + end_time.strftime('%H:%M:%S'))
        print('------------------------------------')
    
    return (end_time - start_time).total_seconds()

###########################################################################################################
# Function to calculate the number of parameters and trainable parameters:

def num_params(estimation_id: str, models: dict, model_assess: dict) -> tuple:
    """
    Function to calculate the number of parameters and trainable parameters.

    :param estimation_id: model identification.
    :type estimation_id: string.
    :param models: collection of models.
    :type models: dict.
    :param model_assess: description of models.
    :type model_assess: dict.

    :return: number of parameters and number of trainable fitted parameters.
    :rtype: tuple.
    """
    if model_assess[estimation_id]['data_modeling']['which_model'] in ['ann', 'cnn', 'transfer']:
        trainable_count = count_params(models[estimation_id].model.trainable_weights)
        non_trainable_count = count_params(models[estimation_id].model.non_trainable_weights)
        return (trainable_count+non_trainable_count, trainable_count)

    if model_assess[estimation_id]['data_modeling']['which_model']=='lr':
        return (len(models[estimation_id].coef_.ravel()) + len(models[estimation_id].intercept_.ravel()), None)

    if model_assess[estimation_id]['data_modeling']['which_model']=='lgb':
        trees = model_assess[estimation_id]['data_modeling']['model_params']['num_iterations']
        depth = model_assess[estimation_id]['data_modeling']['model_params']['max_depth']
        leaves = 31
        return (trees*depth*leaves, None)

###########################################################################################################
# Function that returns all values in a nested dictionary:

def nested_dict_values(d):
    """
    ...
    """
    for v in d.values():
        if isinstance(v, dict):
            yield from nested_dict_values(v)
        else:
            yield v

###########################################################################################################
# Function that returns the class associated to each label:

def label_to_class(classes: list) -> dict:
    """
    Function that returns the class associated to each label.

    :param classes: available classes.
    :type classes: list.

    :return: class of each label.
    :rtype: dictionary.
    """
    class_dict = dict(
        zip(
            [i+1 for i in range(len(classes))], classes
        )
    ) # Dictionary with label by class.

    return class_dict
