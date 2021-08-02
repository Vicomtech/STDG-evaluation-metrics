#import libraries
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import seaborn
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc


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
            scaled_num = self.num_scaler.fit_transform(train_data[self.numerical_columns])

            #return the standardized numerical attributes stacked with the one-hot encoded categorical attributes
            return np.column_stack((scaled_num, scaled_cat))

        else : 

            #standardize numerical attributes
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
            scaled_num = self.num_scaler.transform(test_data[self.numerical_columns])

            #return the standardized numerical attributes stacked with the one-hot encoded categorical attributes
            return np.column_stack((scaled_num, scaled_cat))

        else :
            return self.num_scaler.transform(test_data[self.numerical_columns])


def train_evaluate_model(model_name, x_train, y_train, x_test, y_test) :
    """Train and evaluate a classifier model
    
    Parameters
    ----------
    model_name : string
        the name of the model
    x_train : numpy.ndarray
        training data
    y_train : numpy.ndarray
        training labels
    x_test : numpy.ndarray
        testing data
    y_test : numpy.ndarray
        testing labels

    Returns
    -------
    pandas.core.frame.DataFrame
        a dataframe with the obtained classification metrics
    """

    #initialize the model 
    model = initialize_model(model_name)

    #train the model
    model.fit(x_train, y_train)

    #make predictions
    predictions = model.predict(x_test)

    #compute metrics
    model_results = pd.DataFrame([[model_name,  np.round(accuracy_score(y_test, predictions),4),
                                                np.round(precision_score(y_test, predictions, average='weighted'),4),
                                                np.round(recall_score(y_test, predictions, average='weighted'),4),
                                                np.round(f1_score(y_test, predictions, average='weighted'),4)]],
                                                columns = ['model','accuracy','precision','recall','f1'])

    #return metric values
    return model_results


def initialize_model(model_name) :
    """Get the desired classifier model initialized.
    
    Parameters
    ----------
    model_name : string
        the name of the model

    Returns
    -------
    classifier_model
        the especified classifier model initialized
    """

    #define a dict that initialize each classifier model
    switcher = {'RF' : RandomForestClassifier(n_estimators=100, random_state=9, verbose=True, n_jobs=3),
                'KNN' : KNeighborsClassifier(n_neighbors=10, n_jobs=3),
                'DT' : DecisionTreeClassifier(random_state=9),
                'SVM' : SVC(C=100, max_iter=300, kernel='linear', probability=True, random_state=9, verbose=1),
                'MLP' : MLPClassifier(hidden_layer_sizes=(128,64,32), max_iter=300, random_state=9, verbose=1)}

    #return the desired classifier model
    return switcher.get(model_name, "Invalid model name")