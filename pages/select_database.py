import streamlit as st
import pandas as pd
from scripts.model import ModelEvaluation, KNN, SVM, NaiveBayes
from scripts.data_preprocessing import Preprocessor
import os



dataset = st.sidebar.selectbox("Choose Dataset", ["Breast Cancer Wisconsin"])

if dataset == "Breast Cancer Wisconsin":
    project_path = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(project_path,'..','data', 'data.csv')

    df = pd.read_csv(dataset_path)
    st.session_state.df = df
    st.write("Top 10 rows of the dataset:")
    st.write(df.head(10))

algorithm = st.sidebar.selectbox("Choose Algorithm", ["KNN", "SVM", "Naive Bayes"])
if st.sidebar.button("Train Model"):
    st.session_state.algo = algorithm
    st.switch_page("pages/results.py")