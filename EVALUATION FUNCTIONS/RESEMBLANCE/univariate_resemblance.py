#import libraries
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from math import sqrt
from scipy.spatial import distance
from scipy.stats import chi2_contingency
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns


def basic_stats(df) :

    """Returns a dataframe with the mean and std values of the input dataframe.
    
    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        The dataframe to compute the mean and std values

    Returns
    -------
    pandas.core.frame.DataFrame
        a dataframe with the mean and std values of the dataframe
    """

    #select numerical cols of the data
    numerical_cols = df.select_dtypes(include=['int64']).columns

    #change precision of pandas to 4 decimals
    pd.set_option('precision', 4)

    #get mean and std values dataframe
    means = df.describe().loc['mean'].values
    stds = df.describe().loc['std'].values

    #join means and std values in a dataframe
    df_means_std = pd.DataFrame()
    for i in range(0,len(numerical_cols)) :
        row = []
        string = str(np.round(means[i],2)) + ' ± ' + str(np.round(stds[i],2))
        row.append(string)
        row_data = pd.DataFrame(data=[row])
        df_means_std = df_means_std.append(row_data)
    df_means_std.index=numerical_cols

    #return the created dataframe
    return df_means_std

    
def student_t_tests(real, synthetic) :

    """Performs Student T-tests to compare numerical attributes of real data and synthetic data.
    
    Parameters
    ----------
    real : pandas.core.frame.DataFrame
        The real dataframe

    synthetic : pandas.core.frame.DataFrame
        The synthetic dataframe

    Returns
    -------
    list
        a list the p-values of the statistical tests
    """

    #get list of numerical column names
    num_cols = (real.select_dtypes(include=['int64','float64'])).columns

    #initialize a list to save the p-values of the tests
    p_values = []

    #loop to perform the tests for each attribute
    for c in num_cols :
        _, p = stats.ttest_ind(real[c], synthetic[c])
        p_values.append(p)

    #return the obtained p-values
    return p_values


def mann_whitney_tests(real, synthetic) :

    """Performs Mann Whitney U-Tests to compare numerical attributes of real data and synthetic data.
    
    Parameters
    ----------
    real : pandas.core.frame.DataFrame
        The real dataframe

    synthetic : pandas.core.frame.DataFrame
        The synthetic dataframe

    Returns
    -------
    list
        a list the p-values of the statistical tests
    """

    #get list of numerical column names
    num_cols = (real.select_dtypes(include=['int64','float64'])).columns

    #initialize a list to save the p-values of the tests
    p_values = []

    #loop to perform the tests for each attribute
    for c in num_cols :
        _, p = stats.mannwhitneyu(real[c], synthetic[c])
        p_values.append(p)

    #return the obtained p-values
    return p_values


def ks_tests(real, synthetic) :

    """Performs Kolmogorov Smirnov tests to compare numerical attributes of real data and synthetic data.
    
    Parameters
    ----------
    real : pandas.core.frame.DataFrame
        The real dataframe

    synthetic : pandas.core.frame.DataFrame
        The synthetic dataframe

    Returns
    -------
    list
        a list the p-values of the statistical tests
    """

    #get list of numerical column names
    num_cols = (real.select_dtypes(include=['int64','float64'])).columns

    #initialize a list to save the p-values of the tests
    p_values = []

    #loop to perform the tests for each attribute
    for c in num_cols :
        _, p = stats.ks_2samp(real[c], synthetic[c])
        p_values.append(p)

    #return the obtained p-values
    return p_values


def chi_squared_tests(real, synthetic) :

    """Performs Chi-squared tests to compare categorical attributes of real data and synthetic data.
    
    Parameters
    ----------
    real : pandas.core.frame.DataFrame
        The real dataframe

    synthetic : pandas.core.frame.DataFrame
        The synthetic dataframe

    Returns
    -------
    list
        a list the p-values of the statistical tests
    """

    #get list of categorical column names
    cat_cols = (real.select_dtypes(include=['category'])).columns

    #initialize a list to save the p-values of the tests
    p_values = []

    #loop to perform the tests for each attribute
    for c in cat_cols :
        #create contingency table
        observed = pd.crosstab(real[c], synthetic[c])
        #perform chi-squared test
        _, p, _, _ = chi2_contingency(observed)
        p_values.append(p)

    #return the obtained p-values
    return p_values


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


def euclidean_distances(real, synthetic) :

    """Compute the euclidean distances between real data attributes and synthetic data attributes independently.
    
    Parameters
    ----------
    real : pandas.core.frame.DataFrame
        The real dataframe

    synthetic : pandas.core.frame.DataFrame
        The synthetic dataframe

    Returns
    -------
    list
        a list with the distances values
    """

    #get list of numerical column names
    num_cols = (real.select_dtypes(include=['int64','float64'])).columns

    #initialize a list to save the distances values
    dists = []

    #loop to perform the distances for each attribute
    for c in num_cols :
        dists.append(distance.euclidean(real[c].values, synthetic[c].values))

    #return the list with the computed distances
    return dists


def cosine_distances(real, synthetic) :

    """Compute the cosine distances between real data attributes and synthetic data attributes independently.
    
    Parameters
    ----------
    real : pandas.core.frame.DataFrame
        The real dataframe

    synthetic : pandas.core.frame.DataFrame
        The synthetic dataframe

    Returns
    -------
    list
        a list with the distances values
    """

    #get list of numerical column names
    num_cols = (real.select_dtypes(include=['int64','float64'])).columns

    #initialize a list to save the distances values
    dists = []

    #loop to perform the distances for each attribute
    for c in num_cols :
        dists.append(distance.cosine(real[c].values, synthetic[c].values))

    #return the list with the computed distances
    return dists


def js_distances(real, synthetic) :

    """Compute the Jenshen-Shannon distances between real data attributes and synthetic data attributes independently.
    
    Parameters
    ----------
    real : pandas.core.frame.DataFrame
        The real dataframe

    synthetic : pandas.core.frame.DataFrame
        The synthetic dataframe

    Returns
    -------
    list
        a list with the distances values
    """

    #get list of numerical column names
    num_cols = (real.select_dtypes(include=['int64','float64'])).columns

    #initialize a list to save the distances values
    dists = []

    #loop to perform the distances for each attribute
    for c in num_cols :
        dists.append(distance.jensenshannon(stats.norm.pdf(real[c].values), stats.norm.pdf(synthetic[c].values)))

    #return the list with the computed distances
    return dists


def wass_distances(real, synthetic) :

    """Compute the Wasserstein distances between real data attributes and synthetic data attributes independently.
    
    Parameters
    ----------
    real : pandas.core.frame.DataFrame
        The real dataframe

    synthetic : pandas.core.frame.DataFrame
        The synthetic dataframe

    Returns
    -------
    list
        a list with the distances values
    """

    #get list of numerical column names
    num_cols = (real.select_dtypes(include=['int64','float64'])).columns

    #initialize a list to save the distances values
    dists = []

    #loop to perform the distances for each attribute
    for c in num_cols :
        dists.append(stats.wasserstein_distance(real[c].values, synthetic[c].values))

    #return the list with the computed distances
    return dists