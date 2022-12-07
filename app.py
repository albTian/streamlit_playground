import streamlit as st
import numpy as np
import pandas as pd
from ml import train_model


def getInputOutput(df):
    allInputOptions = []
    allOutputOptions = []
    g = df.columns.to_series().groupby(df.dtypes).groups
    for g_key in g.keys():
        if g_key.name == 'bool':
            allOutputOptions.extend(g[g_key])
        else:
            allInputOptions.extend(g[g_key])
    return allInputOptions, allOutputOptions

def train_model_callback(df, inputOptions, outputOption):
    train_model(df, output=outputOption, inputs=inputOptions)
    st.write("dogwater")

def main():
    csv_file = st.file_uploader("Upload a CSV", type=["csv"])
    if (csv_file):
        # Read the table
        df = pd.read_csv(csv_file)

        # Display the table
        number = st.slider("Display how many rows", 0, len(df))
        st.write(df.head(number))

        # Select input options
        allInputOptions, allOutputOptions = getInputOutput(df)
        inputOptions = st.multiselect("Pick input features",allInputOptions)
        outputOption = st.radio("Pick an output options",allOutputOptions)

        # Train model
        st.button("Train model", on_click=train_model_callback, args=(df, inputOptions, outputOption))

main()