import streamlit as st
import pandas as pd
from scripts.model import ModelEvaluation, KNN, SVM, NaiveBayes
from scripts.data_preprocessing import Preprocessor
import os

df = st.session_state.df

preprocessor = Preprocessor(df)
preprocessor.col_cleanup()
preprocessor.encoder()


st.write("Last 10 rows of the dataset:")
st.write(df.tail(10))

st.write("Correlation Matrix:")
st.pyplot(preprocessor.plot().figure)

X_train, X_test, y_train, y_test = preprocessor.split_test_train()

algorithm = st.session_state.algo

if algorithm == "KNN":
    model = KNN(X_train, y_train)
elif algorithm == "SVM":
    model = SVM(X_train, y_train)
elif algorithm == "Naive Bayes":
    model = NaiveBayes(X_train, y_train)

st.write("Model training...")

model.train()
predictions = model.predict(X_test)

st.warning("Done.")

accuracy, precision, recall, f1 = ModelEvaluation.calculate_metrics(y_test, predictions)
st.write("Metrics:")
st.write(f"Accuracy: {accuracy}")
st.write(f"Precision: {precision}")
st.write(f"Recall: {recall}")
st.write(f"F1-score: {f1}")

st.write("Confusion Matrix:")
st.pyplot(ModelEvaluation.confusion_matrix(y_test, predictions).figure)