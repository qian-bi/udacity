#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Date    : 2017-11-18 13:40:56
@Author  : Liao Jiabin
'''
####################
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
####################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def filter_data(data, condition):
    '''
    Remove element that do not match the condition provided.
    Takes a data list as input and return a filter list.
    Conditions should be a list of string of the following format:
        '<field> <op> <value>'
    where the following operations are valid: >, <, >=, <=, ==, !=

    Example: ["Sex == 'male'", 'Age < 18']
    '''

    field, op, value = condition.split(' ')
    try:
        value = float(value)
    except Exception:
        value = value.strip('\'\"')

    if op == '>':
        matches = data[field] > value
    elif op == '<':
        matches = data[field] < value
    elif op == '>=':
        matches = data[field] >= value
    elif op == '<=':
        matches = data[field] <= value
    elif op == '==':
        matches = data[field] == value
    elif op == '!=':
        matches = data[field] != value
    else:
        raise Exception('Invalid comparison operator. '
                        'Only >, <, >=, <=, ==, != allowed.')

    data = data[matches].reset_index(drop=True)
    return data


def survival_stats(data, outcomes, key, filters=[]):
    '''
    Print out selected statistics regarding survival, given a featurn of
    insterest and any number of filters (including no filters)
    '''

    if key not in data.columns.values:
        print("'{}' is not a feature of the Titanic data. "
              "Did you spell something wrong?".format(key))
        return False
    elif key == 'Cabin' or key == 'PassengerId' or key == 'Ticket':
        print("'{} has too many unique categories to display! "
              "Try a different feature.".format(key))
        return False

    all_data = pd.concat([data, outcomes.to_frame()], axis=1)
    for condition in filters:
        all_data = filter_data(all_data, condition)

    all_data = all_data[[key, 'Survived']]
    plt.figure(figsize=(8, 6))
    if key == 'Age' or key == 'Fare':
        all_data = all_data[~np.isnan(all_data[key])]
        min_value = all_data[key].min()
        max_value = all_data[key].max()
        value_range = max_value - min_value

        if key == 'Fare':
            bins = np.arange(0, max_value + 20, 20)
        else:
            bins = np.arange(0, max_value + 10, 10)

        nonsurv_vals = all_data[all_data['Survived'] == 0][
            key].reset_index(drop=True)
        surv_vals = all_data[all_data['Survived'] == 1][
            key].reset_index(drop=True)
        plt.hist(nonsurv_vals, bins=bins, alpha=0.6,
                 color='red', label='Did not survive')
        plt.hist(surv_vals, bins=bins, alpha=0.6,
                 color='green', label='Survived')
        plt.xlim(0, bins.max())
        plt.legend(framealpha=0.8)

    else:
        if key == 'Pclass':
            values = np.arange(1, 4)
        elif key == 'Parch' or key == 'SibSp':
            values = np.arange(0, np.max(data[key]) + 1)
        elif key == 'Embarked':
            values = ['C', 'Q', 'S']
        elif key == 'Sex':
            values = ['male', 'female']

        frame = pd.DataFrame(index=np.arange(len(values)),
                             columns=(key, 'Survived', 'NSurvived'))
        for i, value in enumerate(values):
            frame.loc[i] = [value,
                            len(all_data[(all_data['Survived'] == 1) &
                                         (all_data[key] == value)]),
                            len(all_data[(all_data['Survived'] == 0) &
                                         (all_data[key] == value)])]

        bar_width = 0.4
        for i in np.arange(len(frame)):
            nonsurv_bar = plt.bar(i - bar_width, frame.loc[i]['NSurvived'],
                                  width=bar_width, color='r')
            surv_bar = plt.bar(
                i, frame.loc[i]['Survived'], width=bar_width, color='g')
            plt.xticks(np.arange(len(frame)), values)
            plt.legend((nonsurv_bar[0], surv_bar[0]),
                       ('Did not survived', 'Survived'), framealpha=0.8)

    plt.xlabel(key)
    plt.ylabel('Number of Passengers')
    plt.title('Passenger Survival Statistics With \'%s\' Feature' % key)

    if sum(pd.isnull(all_data[key])):
        nan_outcomes = all_data[pd.isnull(all_data[key])]['Survived']
        print("Passengers with missing '{}' values: {} ({} survived, {}"
              " did not survive).".format(key, len(nan_outcomes),
                                          sum(nan_outcomes == 1),
                                          sum(nan_outcomes == 0)))
