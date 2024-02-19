###Requirements
  streamlit
  pandas
  seaborn
  matplotlib
  scikit-learn

###How to run?
  Run 'streamlit run main.py' in project directory

###data_preprocessing.py:
  This module has only one class. This class have those following methods:
    col_cleanup: cleans irrevelant columns
    encoder: encodes target column
    plot: returns a correlation matrix
    split_test_train: split dataframe into test and train data

###model.py:
  This module has two classes. Model and it's child classes and ModelEvaluation.
  Model class is a abstract class and should be use for a guide to create child classes.
  Each model class have train and predict methods.

  ModelEvaluation is for performans metrics. calculate_metrics method return accuracy, precision, recall and f1 score. confusion_matrix creates a confusion matrix and returns it.


Streamlit pages are stored in pages directory. main.py is just a redirection to select_database page which is for choosing database and algorithm. When button is pressed model page will be run.
In the background model.py calls both model and data_preprocessing modules to complete project goals.
