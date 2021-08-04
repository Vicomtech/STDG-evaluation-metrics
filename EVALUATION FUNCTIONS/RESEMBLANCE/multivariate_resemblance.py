#import libraries
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import chi2_contingency
import matplotlib.gridspec as gridspec


def get_numerical_correlations(df) :

    """Computes the pairwise pearson correlation matrix and its norm of numerical attributes of a dataframe.
    
    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        The dataframe to compute the pairwise pearson correlation matrix

    Returns
    -------
    pandas.core.frame.DataFrame
        a dataframe with the pairwise pearson correlation values of the numerical attributes of the dataframe

    numpy.float64
        the norm of the pairwise pearson correlation matrix of the numerical attributes of the dataframe
    """

    #compute the pearson pairwise correlation matrix of numerical attributes of the dataset
    cors = np.absolute(df.corr(method='pearson'))

    #compute the norm of the pearson pairwise correlation matrix computed before
    cors_norm = np.round(np.linalg.norm(cors),4)

    #return the values
    return cors, cors_norm


def plot_correlations(cors, ax_plot, color_bar) :

    """Plot a pairwise pearson correlation matrix.
    
    Parameters
    ----------
    cors : pandas.core.frame.DataFrame
        A dataframe with the pairwise pearson correlation matrix

    ax_plot : matplotlib.axes._subplots.AxesSubplot
        Axes to plot the correlation matrix

    color_bar : bool
        Boolean to indicate whether to show the color bar or not
    """

    #delete redundancy of the corelation matrix
    cors = cors.iloc[1:, 0:-1]
    
    #compute the mask of the correlation matrix to plot only one side of it
    cors_mask = np.triu(np.ones_like(cors, dtype=bool)) - np.identity(len(cors))

    #plot a heatmap with the correlation matrix values
    sns.heatmap(cors,  linewidths=.3, ax=ax_plot, mask=cors_mask, cbar=color_bar, vmin=0, vmax=1, cmap='Blues')


def get_categorical_correlations(df) :

    """Computes the normalized contingency table and its norm of categorical attributes of a dataframe.
    
    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        The dataframe to compute the normalized contingency table

    Returns
    -------
    pandas.core.frame.DataFrame
        a dataframe with the normalized contingency table of the categorical attributes of the dataframe

    numpy.float64
        the norm of the normalized contingency table of the categorical attributes of the dataframe
    """

    #select categorical attributes of the dataframe
    df = df.select_dtypes(include=['category'])

    #get each combination of attributes pairs.
    factors_paired = [(i,j) for i in df.columns.values for j in df.columns.values] 

    #initialize lists to save the chi2 and the p values
    chi2, p_values =[], []

    #loop to iterate over each attributes pair
    for f in factors_paired:

        #compute the contingency table of the attributes pair
        if f[0] != f[1]: #for different factor pair
            chitest = chi2_contingency(pd.crosstab(df[f[0]], df[f[1]]))   
            chi2.append(chitest[0])
            p_values.append(chitest[1])
        else:   #for same factor pair
            chi2.append(0)
            p_values.append(0)

    #save the contingency table as a dataframe
    chi2 = np.array(chi2).reshape((df.shape[1],df.shape[1])) # shape it as a matrix
    chi2 = pd.DataFrame(chi2, index=df.columns.values, columns=df.columns.values) # then a df for convenience

    #normalize the contingency table
    normalized_chi2 = (chi2 - np.min(chi2))/np.ptp(chi2)

    #calculate the norm of the normalized contingency table
    norm = np.round(np.linalg.norm(normalized_chi2),4)

    #return the values
    return normalized_chi2, norm


def compute_mra_score(real, synthetic) :
    """Computes the percentage of correlations that are preserved in synthetic data.
    
    Parameters
    ----------
    real : pandas.core.frame.DataFrame
        The dataframe with the correlations of real data
        
    synthetic : pandas.core.frame.DataFrame
        The dataframe with the correlations of synthetic data

    Returns
    -------
    numpy.float64
        a value that indicates the percentage of correlations that are preserved in synthetic data
    """
    
    #get the correlations differences between real data and synthetic data
    diff = abs(real - synthetic)
    diffs = diff.values[np.triu_indices(len(diff),k=1)]

    #compute the percentage of preserved correlations
    total_cors = len(diffs)
    preserved_cors = len(diffs[diffs < 0.1])
    
    #return the percentage of correlations preserved in synthetic data (rounded to two decimals)
    return np.round(preserved_cors/total_cors,2)