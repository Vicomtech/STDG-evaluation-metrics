  
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder


class DataPreProcessor :
    """
    A class used to preprocess train data and transform generated data from GM approach

    ...

    Attributes
    ----------
    int_features : list
        a list with the integer columns names of the dataframe
    float_features : list
        a list with the float columns names of the dataframe
    categorical_columns : list
        a list with the categorical columns names of the dataframe
    train_data : pandas.core.frame.DataFrame
        a dataframe that contains the data to be preprocessed for training GM approach
    categorical_encoders : dict
        a dictionary that contains one label encoder per each categorical attribute
    one_hot_encoders : dict
        a dictionary that contains one one-hot encoder per each categorical attribute
   encoded_vars : dict
        a dictionary that contains the value of the encoded categorical attributes

    Methods
    ----------
    preprocess_train_data()
        Preprocess the train data
    transform_data(generated_samples)
        Tranforms the generated samples according to the encoders trained with train data to obtain synthetic data
    """

    def __init__(self, data) :
        """
        Parameters
        ----------
        data : pandas.core.frame.DataFrame
            data to be preprocessed
        """
        
        #initialize the list of int, float and categorical attributes
        self.int_features = data.select_dtypes(include=['int64']).columns.tolist()
        self.float_features = data.select_dtypes(include=['float64']).columns.tolist()
        self.categorical_columns = data.select_dtypes(include=['category']).columns.tolist()
        self.train_data = data

        #initialize the dictionaries that will contain the encoders for each categorical attribute
        self.categorical_encoders = dict()
        self.one_hot_encoders = dict()
        self.encoded_vars = dict()
        self.scaler = StandardScaler()
    

    def preprocess_train_data(self) :
        """Preprocess the train data

        Returns
        -------
        pandas.core.frame.DataFrame
            a dataframe with the preprocessed data
        """

        #copy the train data
        data = self.train_data.copy()

        #iterate over all categorical columns
        for column in self.categorical_columns :

            #fit one-hot encoder for the attribute and transform it
            self.one_hot_encoders[column] = OneHotEncoder().fit(np.asarray(data[column].astype('category')).reshape(-1,1))
            self.encoded_vars[column] = self.one_hot_encoders[column].transform(np.asarray(data[column].astype('category')).reshape(-1,1)).toarray()
            data = data.drop([column], axis=1)

            #compute the inverse sigmoid function for each one-hot encoded column
            for i in range(0,self.encoded_vars[column].shape[1]) :
                data[column + str(i)] = self.encoded_vars[column][:,i]
                data[column + str(i)] = data[column + str(i)].astype('int8')
                data[column + str(i)] = np.exp(np.asarray(data[column + str(i)].values)) / (1 + np.exp(np.asarray(data[column + str(i)].values)))

        #standardize the data
        scaled_data = self.scaler.fit_transform(data.values)

        #return the preprocessed data
        return pd.DataFrame(scaled_data, columns = data.columns.tolist())


    def transform_data(self, generated_samples) :
        """Tranforms the generated samples according to the encoders trained with train data to obtain synthetic data
    
        Parameters
        ----------
        generated_samples : pandas.core.frame.DataFrame
            a dataframe with the generated data to be transformed

        Returns
        -------
        pandas.core.frame.DataFrame
            a dataframe with the transformed data
        """

        #inverse standardization of data
        generated_samples[generated_samples.columns] = self.scaler.inverse_transform(generated_samples[generated_samples.columns])

        #convert the integer attributes of the dataframe
        for c in self.int_features :
            generated_samples[c] = generated_samples[c].astype('int64')
            synthetic_data = generated_samples.select_dtypes(include=['int64'])

        #convert the float attributes of the dataframe
        for c in self.float_features :
            generated_samples[c] = generated_samples[c].astype('float64')
            synthetic_data = generated_samples.select_dtypes(include=['float64'])

        #transform categorical features to original features types
        for col in self.categorical_columns : 

            #get the obtained numerical values of each categorical attribute encoded group
            cols_drop = (generated_samples.filter(regex=col)).columns.tolist()
            values = generated_samples.filter(regex=col).values
            generated_samples = generated_samples.drop(cols_drop, axis = 1)

            #iterate over all values of assign a 1 to the maximum value row, to the rest it gives a value of 0
            for i in range(0,values.shape[0]) :
                m = max(values[i,:])
                for j, k in enumerate(values[i,:]) :
                    if k == m :
                        values[i,j] = 1
                    else :
                        values[i,j] = 0

            #perform the inverse one-hot encoding of the categorical attribute
            generated_samples[col] = self.one_hot_encoders[col].inverse_transform(values)

        #sort the attributes of the dataframe to be in the same order as in real train data
        synthetic_data = pd.DataFrame(columns = self.train_data.columns)
        for col in self.train_data.columns :
            synthetic_data[col] = generated_samples[col]

        #return the transformed synthetic dataframe
        return synthetic_data