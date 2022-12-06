import streamlit as st
import numpy as np
import pandas as pd


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


csv_file = st.file_uploader("Upload a CSV", type=["csv"])
if (csv_file):
    df = pd.read_csv(csv_file)
    number = st.slider("Display how many rows", 0, len(df))
    st.write(df.head(number))
    allInputOptions, allOutputOptions = getInputOutput(df)
    inputOptions = st.multiselect("Pick input features",allInputOptions)
    outputOptions = st.radio("Pick an output options",allOutputOptions)
    # st.write(allInputOptions)
    # st.write(allOutputOptions)
