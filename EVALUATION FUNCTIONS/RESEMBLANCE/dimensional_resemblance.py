#import libraries
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.manifold import Isomap
from umap import UMAP
import matplotlib.patches as mpatches


def preprocess_data(data) :

    """Preprocess the given dataset applying One-Hot encoding of categorical attributes and Standardization for numerical attributes.
    
    Parameters
    ----------
    data : pandas.core.frame.DataFrame
        The dataframe to be preprocessed

    Returns
    -------
    numpy.ndarray
        a matrix with the preprocessed data
    """

    #get categorical attributes
    categorical_vars = data.select_dtypes(include=['category'])

    #label encoder each categorical attribute
    for c in categorical_vars.columns.tolist() :
        categorical_vars[c] = LabelEncoder().fit_transform(categorical_vars[c])

    #one-hot encode categorical variables
    onehot_encoder_x = OneHotEncoder()
    x_cat = onehot_encoder_x.fit_transform(categorical_vars).toarray()

    #standardize numerical variables
    numerical_vars =  data.select_dtypes(include=['int64','float64'])
    x_num = StandardScaler().fit_transform(numerical_vars)

    #return the standardized numerical attributes stacked with the one-hot encoded categorical attributes
    return np.column_stack((x_num, x_cat))


def pca_transform(data_scaled, labels) :

    """Compute the PCA transform from a scaled data.
    
    Parameters
    ----------
    data_scaled : numpy.ndarray
        A matrix with the scaled data to transform

    labels : numpy.ndarray
        Labels to assign to the transformed data (0 for real and 1 for synthetic)

    Returns
    -------
    pandas.core.frame.DataFrame
        a dataframe with the transformed data labelled
    """

    #compute the PCA transform
    pca_transform = PCA(n_components=2).fit_transform(data_scaled)

    #append labels to the transformed data
    pca = np.append(pca_transform, labels, axis=1)

    #return a dataframe with the transformed data
    return pd.DataFrame(data=pca, columns=['PC1','PC2','Label'])


def isomap_transform(data_scaled, labels) :

    """Compute the Isomap transform from a scaled data.
    
    Parameters
    ----------
    data_scaled : numpy.ndarray
        A matrix with the scaled data to transform

    labels : numpy.ndarray
        Labels to assign to the transformed data (0 for real and 1 for synthetic)

    Returns
    -------
    pandas.core.frame.DataFrame
        a dataframe with the transformed data labelled
    """

    #compute the Isomap transform
    iso_transform = Isomap(n_components=2).fit_transform(data_scaled)

    #append labels to the transformed data
    iso = np.append(iso_transform, labels, axis=1)

    #return a dataframe with the transformed data
    return pd.DataFrame(data=iso, columns=['PC1','PC2','Label'])


def batch(iterable, n=1) :

    """Create iterable batches from a dataframe.
    
    Parameters
    ----------
    iterable : numpy.ndarray
        A matrix to be divided in batches

    n : int
        Length of the batches to create
    """

    #get length of the matrix to divide in batches
    l = len(iterable)

    #loop to divide the data into batches of length n
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def isomap_transform_on_batch(data_scaled, labels) :

    """Compute the Isomap transform on batch from a scaled data.
    
    Parameters
    ----------
    data_scaled : numpy.ndarray
        A matrix with the scaled data to transform

    labels : numpy.ndarray
        Labels to assign to the transformed data (0 for real and 1 for synthetic)

    Returns
    -------
    pandas.core.frame.DataFrame
        a dataframe with the transformed data labelled
    """

    #initialize dataframe to save the values of the transformation
    iso_df = pd.DataFrame(columns=['PC1','PC2','Label'])
        
    #loop to iterate over all batches of data
    for (b, y) in zip(batch(data_scaled,10000),batch(labels,10000)) :

        #transform the batch of data
        iso_transform = Isomap(n_components=2).fit_transform(b)

        #append the labels of the data
        iso = np.append(iso_transform, y, axis=1)

        #append the transformation of the actual batch to the dataframe that contains the transformation of all the batches
        iso = pd.DataFrame(data=iso, columns=['PC1','PC2','Label'])
        iso_df = iso_df.append(iso, ignore_index=True)

    #return a dataframe with the transformed data
    return iso_df


def umap_transform(data_scaled, labels) :

    """Compute the UMAP transform from a scaled data.
    
    Parameters
    ----------
    data_scaled : numpy.ndarray
        A matrix with the scaled data to transform

    labels : numpy.ndarray
        Labels to assign to the transformed data (0 for real and 1 for synthetic)

    Returns
    -------
    pandas.core.frame.DataFrame
        a dataframe with the transformed data labelled
    """

    #compute the UMAP transform
    umap_transform = UMAP().fit_transform(data_scaled)

    #append labels to the transformed data
    umap = np.append(umap_transform, labels, axis=1)

    #return a dataframe with the transformed data
    return pd.DataFrame(data=umap, columns=['PC1','PC2','Label'])


def umap_transform_on_batch(data_scaled, labels) :

    """Compute the UMAP transform on batch from a scaled data.
    
    Parameters
    ----------
    data_scaled : numpy.ndarray
        A matrix with the scaled data to transform

    labels : numpy.ndarray
        Labels to assign to the transformed data (0 for real and 1 for synthetic)

    Returns
    -------
    pandas.core.frame.DataFrame
        a dataframe with the transformed data labelled
    """

    #initialize dataframe to save the values of the transformation
    umap_df = pd.DataFrame(columns=['PC1','PC2','Label'])

    #loop to iterate over all batches of data
    for (b, y) in zip(batch(data_scaled,10000),batch(labels,10000)) :

        #transform the batch of data
        umap_transform = UMAP().fit_transform(b)

        #append the labels of the data
        umap_new = np.append(umap_transform, y, axis=1)

        #append the transformation of the actual batch to the dataframe that contains the transformation of all the batches
        umap_new = pd.DataFrame(data=umap_new, columns=['PC1','PC2','Label'])
        umap_df = umap_df.append(umap_new, ignore_index=True)

    #return a dataframe with the transformed data
    return umap_df

