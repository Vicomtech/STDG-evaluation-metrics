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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc


def mix_data(real, synthetic) :
    """Mix a real and a synthetic dataframe in one unique dataframe with labelled data (0 for real and 1 for synthetic)
    
    Parameters
    ----------
    real : pandas.core.frame.DataFrame
        Real dataframe

    synthetic : numpy.ndarray
        Synthetic dataframe

    Returns
    -------
    pandas.core.frame.DataFrame
        a dataframe with the real and synthetic data mixed
    """

    #add labels to real (0) and synthetic (1) data
    real['Label'] = np.zeros(real.shape[0]).astype('int8')
    synthetic['Label'] = np.ones(synthetic.shape[0])

    #mix real and synthetic records
    frames = [real, synthetic]

    #return the concatenate dataframe with the mixed samples
    return pd.concat(frames).sample(frac=1)


def split_data(data, train_len) :
    """Split a dataframe into train and test data
    
    Parameters
    ----------
    data : pandas.core.frame.DataFrame
        Dataframe to split

    train_len : float
        Percentage of data for training

    Returns
    -------
    pandas.core.frame.DataFrame
        a dataframe with the train data

    pandas.core.frame.DataFrame
        a dataframe with the test data
    """

    #calculate the index on which to split the data
    train_split = int(len(data)*train_len)

    #split the data into train and test
    train_data = data[0:train_split]
    test_data = data[train_split+1:len(data)]

    #return the splitted data
    return train_data, test_data


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

        #one-hot encode categorical attributes
        categorical_vars = train_data[self.categorical_columns]
        for col in categorical_vars.columns :
            categorical_vars[col] = categorical_vars[col].astype('category').cat.codes
        scaled_cat = self.onehot_encoder.fit_transform(categorical_vars).toarray()

        #standardize numerical attributes
        scaled_num = self.num_scaler.fit_transform(train_data[self.numerical_columns])

        #return the standardized numerical attributes stacked with the one-hot encoded categorical attributes
        return np.column_stack((scaled_num, scaled_cat))
    
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

        #one-hot encode categorical variables
        categorical_vars = test_data[self.categorical_columns]
        for col in categorical_vars.columns :
            categorical_vars[col] = categorical_vars[col].astype('category').cat.codes
        scaled_cat = self.onehot_encoder.transform(categorical_vars).toarray()

        #standardize numerical attributes
        scaled_num = self.num_scaler.transform(test_data[self.numerical_columns])

        #return the standardized numerical attributes stacked with the one-hot encoded categorical attributes
        return np.column_stack((scaled_num, scaled_cat))


class ClassificationModels :
    """
    A class used to train and evaluate five ML classification models to label records in real and synthetic

    ...

    Attributes
    ----------
    classifiers_names : list
        a list with the names of the classifiers
    classifiers : dic
        a list with the classifiers
    resulst : pandas.core.frame.DataFrame
        a dataframe to save the classification metrics results of all the classifiers

    Methods
    ----------
    train_classifiers(x_train, y_train)
        Train the classifiers to label data in real and synthetic.
    evaluate_classifiers(x_test, y_test)
        Evaluate the classifiers when labelling records in real and synthetic.
    plot_classification_metrics(ax_plot)
        Create a boxplot with the obtained classification metrics when evaluating the models.
    """

    def __init__(self) :

        #list of the names of the classifiers
        self.classifiers_names = ['RF','KNN','DT','SVM','MLP']

        #dict with the five classifiers
        self.classifiers = {'RF' : RandomForestClassifier(n_estimators=100, random_state=9, n_jobs=3),
                            'KNN' : KNeighborsClassifier(n_neighbors=10, n_jobs=3),
                            'DT' : DecisionTreeClassifier(random_state=9),
                            'SVM' : SVC(C=100, max_iter=300, kernel='linear', probability=True, random_state=9),
                            'MLP' : MLPClassifier(hidden_layer_sizes=(128,64,32), max_iter=300, random_state=9)}

        #dataframe to save the results of the classifiers
        self.results = pd.DataFrame(columns = ['model','accuracy','precision','recall','f1'])

    def train_classifiers(self, x_train, y_train) :
        """Train the classifiers to label data in real and synthetic.
    
        Parameters
        ----------
        x_train : numpy.ndarray
            Data to train the classifiers
        y_train : numpy.ndarray
            Labels of the train data
        """

        #loop to iterate over all the classifiers
        for clas in self.classifiers_names :

            #fit the classifier
            self.classifiers[clas].fit(x_train, y_train)
            print(clas + ' Trained')

    def evaluate_classifiers(self, x_test, y_test) :
        """Evaluate the classifiers when labelling records in real and synthetic.
    
        Parameters
        ----------
        x_test : numpy.ndarray
            Data to evaluate the classifiers
        y_test : numpy.ndarray
            Real labels of the test data
        """

        #loop to iterate over all classifiers
        for clas in self.classifiers_names :
           
            #make predictions
            predictions = self.classifiers[clas].predict(x_test)

            #compute metrics
            clas_results = pd.DataFrame([[clas,np.round(accuracy_score(y_test, predictions),4),
                                        np.round(precision_score(y_test, predictions),4),
                                        np.round(recall_score(y_test, predictions),4),
                                        np.round(f1_score(y_test, predictions),4)]],
                                        columns = self.results.columns)

            #append to results dataframe
            self.results = self.results.append(clas_results, ignore_index=True)
            print(clas + ' Tested')
            print(clas_results)

    def plot_classification_metrics(self, ax_plot) :
        """Create a boxplot with the obtained classification metrics when evaluating the models.
    
        Parameters
        ----------
        ax_plot : matplotlib.axes._subplots.AxesSubplot
            Axes in where to create the boxplot
        """

        #define lists with the metrics
        metrics = ['accuracy','precision','recall','f1']
        metrics_res = ['acc','prec','rec','f1']

        #get data of the boxplot
        boxplot_data = self.results[metrics]

        #create and edit the boxplot
        ax_plot.boxplot(boxplot_data.values)
        ax_plot.set_xticks([1,2,3,4])
        ax_plot.set_xticklabels(metrics_res)
        ax_plot.set_ylim(bottom=-0.05, top=1.05)

        