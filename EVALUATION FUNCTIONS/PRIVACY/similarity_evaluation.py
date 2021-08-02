#import libraries
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
import scipy
from math import sqrt
from scipy.spatial import distance
from scipy.stats import chi2_contingency
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity


def scale_data(df) :
    """Scale a dataframe to get the values between 0 and 1. It returns the scaled dataframe.
    
    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        The dataframe to scale

    Returns
    -------
    pandas.core.frame.DataFrame
        A dataframe with the scaled data
    """

    #initialize and fit the scaler
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    #return the scaled dataframe
    return pd.DataFrame(scaled, columns=df.columns.tolist())


def pairwise_euclidean_distance(synthetic_data, real_data) :
    """Compute the pairwise euclidean distances between each pair of real and synthetic records
    
    Parameters
    ----------
    synthetic_data : numpy.ndarray
        Synthetic data records
    real_data : numpy.ndarray
        Real data records

    Returns
    -------
    string
        a string with the mean and std values of the computed pairwise euclidean distances
    """

    #compute the pairwise euclidean distances
    distances = distance.cdist(synthetic_data, real_data, 'euclidean')

    #return the mean and std values of the computed pairwise euclidean distances
    return str(np.round(np.mean(distances),4)) + ' ± ' + str(np.round(np.std(distances),4))


def hausdorff_distance(synthetic_data, real_data) :
    """Compute the hausdorff distance between synthetic dataset and real dataset
    
    Parameters
    ----------
    synthetic_data : numpy.ndarray
        Synthetic data records
    real_data : numpy.ndarray
        Real data records

    Returns
    -------
    float
        the hausdorf distance between synthetic and real datasets
    """

    #compute the hausdorff distance
    hausdorff_dist = scipy.spatial.distance.directed_hausdorff(synthetic_data, real_data)[0]

    #return the computed value rounded on 4 decimals
    return np.round(hausdorff_dist,4)


def rts_similarity(synthetic_data, real_data) :
    """Compute the real to synthetic similarity between each pair of synthetic and real records
    
    Parameters
    ----------
    synthetic_data : numpy.ndarray
        Synthetic data records
    real_data : numpy.ndarray
        Real data records

    Returns
    -------
    dict
        a dictionary with the min, mean and max values of the computed similarity values
    """

    #compute the cosine similarity between each pair of synthetic and real records
    str_sim = cosine_similarity(synthetic_data, real_data)

    #return a dictionary with the min, mean and max values of the computed similarity values
    return {'min' : np.round(np.min(str_sim),4), 'mean' : np.round(np.mean(str_sim),4), 'max' : np.round(np.max(str_sim),4)}