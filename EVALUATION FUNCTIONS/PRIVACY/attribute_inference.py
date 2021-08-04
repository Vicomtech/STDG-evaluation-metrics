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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from itertools import permutations 
from itertools import combinations
from math import sqrt


class RiskAttributesPredictors :
    """
    A class used to train and evaluate Decision Tree predictive models to predict the values of risk attributes based on QIDs.

    ...

    Attributes
    ----------
    qid_columns : list
        a list with the names of the QIDs attributes names
    risk_attributes : list
        a list with the names of the predictive risk attributes names
    data_risk : pandas.core.frame.DataFrame
        a dataframe that contains the risk data
    attributes_models : dict
        a dictionary that contain the predictive models for all the risk attributes

    Methods
    ----------
    train_attributes_prediction_models(self, x_train)
        Train the prediction models for the risk attributes
    evaluate_attributes_prediction_models(self, x_test, batch, columns_results)
        Evaluate the prediction models for the risk attributes
    """

    def __init__(self, data_risk, qid_columns) :

        #list of the names of QID attributes
        self.qid_columns = qid_columns

        #list of the names of risk attributes
        self.risk_attributes = data_risk.columns.tolist()

        #risk data
        self.data_risk = data_risk

        #create a dict for the predictive models for the risk attributes
        self.attributes_models = dict()

        #initialize the predictive models
        for attribute in self.risk_attributes :
            if np.dtype(self.data_risk[attribute]) == 'float64' :
                self.attributes_models[attribute] = DecisionTreeRegressor(random_state=64)
            else :
                self.attributes_models[attribute] = DecisionTreeClassifier(random_state=64)


    def train_attributes_prediction_models(self, x_train) :  
        """Train the prediction models for the risk attributes
    
        Parameters
        ----------
        x_train : numpy.ndarray
            training data
        """    

        #iterate over all risk attributes list
        for attribute in self.risk_attributes :

            #get the output of the models (the values of the actual risk attribute)
            y_train = self.data_risk[attribute]
        
            #train the model for the actual risk attribute
            self.attributes_models[attribute].fit(x_train, y_train)
            print('Model trained for', attribute, 'attribute')


    def evaluate_attributes_prediction_models(self, x_test, batch, columns_results) : 
        """Evaluate the prediction models for the risk attributes
    
        Parameters
        ----------
        x_test : numpy.ndarray
            testing data
        batch : numpy.ndarray
            batch of QIDs
        columns_results : list
            a list the column names of the results dataframe

         Returns
        -------
        pandas.core.frame.DataFrame
            a dataframe with the obtained results from the models evaluation
        """    

        #initialize row data
        row_data = (batch[self.qid_columns].values[0]).tolist()

        #iterate over all risk attributes
        for attribute in self.risk_attributes :

            #get real values and predictions of the model
            y_test = batch[attribute]
            predictions = self.attributes_models[attribute].predict(x_test)

            #calculate the prediction metric
            if np.dtype(y_test) == 'float64' :
                result = sqrt(mean_squared_error(y_test, predictions))
            else :
                result = accuracy_score(y_test.astype('category').cat.codes, predictions)
            print('Model evaluated for', attribute, 'attribute')

            #append the prediction metric to the data of the row
            row_data.append(result)

        #return the dataframe with the obtained results from the models evaluation
        return pd.DataFrame([row_data], columns = columns_results)

        
class DataPreProcessor :
    """
    A class used to preprocess train and test data.

    ...

    Attributes
    ----------
    numerical_columns : list
        a list with the numerical columns names of the dataframe
    categorical_columns : list
        a list with the categorical columns names of the dataframe
    categories : list
        a list of arrays especifying the categories of each categorical attribute to one-hot encode categorical attributes
    label_encoders : list
        a list of LabelEncoders for the categorical attributes
    num_scaler : sklearn.preprocessing.StandardScaler
        a StandardScaler to scale numerical attributes
    onehot_encoder : sklearn.preprocessing.OneHotEncoder
        a Encoder to One-Hot encode categorical attributes

    Methods
    ----------
    preprocess_train_data(train_data)
        Preprocess the train data
    preprocess_test_data(test_data)
        Preprocess the test data
    """

    def __init__(self, categorical_columns, numerical_columns, categories) :
        """
        Parameters
        ----------
        categorical_columns : list
            a list with the categorical columns names of the dataframe
        numerical_columns : list
            a list with the numerical columns names of the dataframe
        categories : list
            a list of arrays especifying the categories of each categorical attribute to one-hot encode categorical attributes
        """
        
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.label_encoders = dict()
        self.num_scaler = StandardScaler()
        self.onehot_encoder = OneHotEncoder(categories=categories)
    
    def preprocess_train_data(self, train_data) :
        """Preprocess the train data
    
        Parameters
        ----------
        train_data : pandas.core.frame.DataFrame
            the train dataframe to be preprocessed

        Returns
        -------
        numpy.ndarray
            a matrix with the preprocessed data
        """

        if self.categorical_columns is not None :
            #one-hot encode categorical attributes
            categorical_vars = train_data[self.categorical_columns]
            for col in categorical_vars.columns :
                categorical_vars[col] = categorical_vars[col].astype('category').cat.codes
            scaled_cat = self.onehot_encoder.fit_transform(categorical_vars).toarray()

            #standardize numerical attributes
            if self.numerical_columns is not None :
                scaled_num = self.num_scaler.fit_transform(train_data[self.numerical_columns])
                return np.column_stack((scaled_num, scaled_cat))
            else :
                return scaled_cat

        else :
            return self.num_scaler.fit_transform(train_data[self.numerical_columns])

    
    def preprocess_test_data(self, test_data) :
        """Preprocess the test data
    
        Parameters
        ----------
        test_data : pandas.core.frame.DataFrame
            the test dataframe to be preprocessed

        Returns
        -------
        numpy.ndarray
            a matrix with the preprocessed data
        """

        if self.categorical_columns is not None :
            #one-hot encode categorical variables
            categorical_vars = test_data[self.categorical_columns]
            for col in categorical_vars.columns :
                categorical_vars[col] = categorical_vars[col].astype('category').cat.codes
            scaled_cat = self.onehot_encoder.transform(categorical_vars).toarray()

            #standardize numerical attributes
            if self.numerical_columns is not None :
                scaled_num = self.num_scaler.transform(test_data[self.numerical_columns])
                return np.column_stack((scaled_num, scaled_cat))
            else :
                return scaled_cat

        else :
            return self.num_scaler.transform(test_data[self.numerical_columns])


def identified_attributes_percentage(results_data, results_columns) :
    """Calculate the percentage of identified attributes
    
    Parameters
    ----------
    results_data : pandas.core.frame.DataFrame
        the results of the AIA simulation

    results_columns : list
        list of column names for the risk attributes
    
    Returns
    -------
    float
        the percentage of identified attributes
    """
    
    #identify the attributes that have been correctly identified
    results = results_data[results_columns].mode().transpose()[0]
    for c in results_columns :
        if 'rmse' in c :
            results[c] = True if results[c] == 0 else False
        else :
            results[c] = True if results[c] == 1 else False
                
    #compute the proportion of correctly identified attributes
    return np.round(np.sum(results)/len(results),2)