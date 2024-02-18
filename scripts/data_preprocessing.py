import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class Preprocessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def col_cleanup(self):
        if "id" in self.df.columns and "Unnamed: 32" in self.df.columns:
            self.df.drop(["id", "Unnamed: 32"], axis=1, inplace=True)

    def encoder(self):
        self.df["diagnosis"] = self.df["diagnosis"].apply(lambda x: 1 if x == 'M' else 0)

    def plot(self):
        ax = sns.scatterplot(x='radius_mean', y='texture_mean', hue='diagnosis', data=self.df)

        ax.set_title('Correlation Matrix')
        ax.set_xlabel('radius_mean')
        ax.set_ylabel('texture_mean')
        ax.legend(title='Diagnosis', loc='upper right', labels={1: 'kotu', 0: 'iyi'})

        return ax

    def split_test_train(self) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
        y = self.df['diagnosis']
        X = self.df.drop('diagnosis', axis=1)
        return train_test_split(X, y, test_size=0.2)
