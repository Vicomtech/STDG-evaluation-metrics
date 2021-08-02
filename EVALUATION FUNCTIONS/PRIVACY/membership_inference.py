#import libraries
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
import scipy
from math import sqrt
import random
from scipy.spatial import distance
from sklearn.metrics import confusion_matrix
import seaborn
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler


def identified_record(record_values, synthetic_data, th) :
    """Returns if a synthetic record can be identified in the subset of real data of the attacker.
    
    Parameters
    ----------
    record_values : numpy.ndarray
        array of the record that the attacker wants to identify
    synthetic_data : pandas.core.frame.DataFrame
        synthetic dataframe
    th : float
        the threshold to consider that the records has been identified in real data

    Returns
    -------
    bool
        a boolean that indicates if a synthetic record can be identified in real data
    """    

    #calculate the hamming distances
    distances = distance.cdist(np.reshape(record_values, (1, -1)), synthetic_data, metric='hamming')

    #return if there is any distance value lower than the specified threshold
    return (np.array(distances) < th).any()


def get_true_labels(train_data_indexes, attacker_data_indexes) :
    """Returns the true labels of the attacker data records
    
    Parameters
    ----------
    train_data_indexes : numpy.ndarray
        indexes of the real data used for training the STDG approach
    attacker_data_indexes : numpy.ndarray
        indexes of the data that the attacker obtained

    Returns
    -------
    list
        the true labels of the records (1 belong to training set, 0 does not belong to training set)
    """   

    #initialize a list to append the labels
    true_labels = []

    #iterate over all attacker data indexes to append the true label of each record to the list
    for idx in attacker_data_indexes :
        if idx in train_data_indexes :
            true_labels.append(1)
        else :
            true_labels.append(0)

    #return the list with the true labels of the records
    return true_labels


def predict_labels(attacker_data, synthetic_data, th) :
    """Predicts if the attacker data records have been used for training the STDG approach
    
    Parameters
    ----------
    attacker_data : pandas.core.frame.DataFrame
        dataframe of real records that has the attacker
    synthetic_data : pandas.core.frame.DataFrame
        synthetic dataframe
    th : float
        the threshold to consider that the records has been identified in real data

    Returns
    -------
    list
        the predicted labels of the records (1 belong to training set, 0 does not belong to training set)
    """   

    #initialize a list to append the predicted labels
    predicted_labels = []

    #iterate over all attacker data indexes to append the predicted label of each record to the list
    for idx in attacker_data.index.tolist() :
        identified = identified_record(attacker_data.loc[idx].values, synthetic_data, th)
        if  identified :
            predicted_labels.append(1)
        else :
            predicted_labels.append(0)

    #return the list with the true labels of the records
    return predicted_labels


def evaluate_membership_attack(attacker_data, train_data_indexes, synthetic_data, th) :
    """Evaluates the results of the membership inference attack
    
    Parameters
    ----------
    attacker_data : pandas.core.frame.DataFrame
        dataframe of real records that has the attacker
    train_data_indexes : list
        a list with the indexes of the real records used for training the STDG approach
    synthetic_data : pandas.core.frame.DataFrame
        synthetic dataframe
    th : float
        the threshold to consider that the records has been identified in real data

    Returns
    -------
    list
        a list with the precision values of the simulation
    list
        a list with the accuracy values of the simulation
    """   

    #get the true labels of the attacker data records
    true_labels = get_true_labels(train_data_indexes, attacker_data.index.tolist())

    #predict the labels of the attacker data records
    predicted_labels = predict_labels(attacker_data, synthetic_data, th)

    #calculate the precision and accuracy values of the simulation
    precision_values = precision_score(true_labels, predicted_labels)
    accuracy_values = accuracy_score(true_labels, predicted_labels)

    #return the precision and accuracy values
    return precision_values, accuracy_values