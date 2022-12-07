import streamlit as st
import numpy as np
import pandas as pd
from ml import SimpleModel

# For now, only bool is an acceptable output. Everything else is an input


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


def main():
    csv_file = st.file_uploader("Upload a CSV", type=["csv"])
    if (not csv_file):
        return

    # Read the table
    df = pd.read_csv(csv_file)

    # Display the table
    number = st.slider("Display how many rows", 0, len(df))
    st.write(df.head(number))

    # Select input options
    allInputOptions, allOutputOptions = getInputOutput(df)
    inputOptions = st.multiselect("Pick input features", allInputOptions)
    outputOption = st.radio("Pick an output options", allOutputOptions)

    # Train model
    if (len(inputOptions) < 1):
        return

    # Add button to train model here...
    model = SimpleModel(outputOption, inputOptions)
    model.train_model(df)

    # Display hypothetical inputs
    input_dict = {}
    for inputOption in inputOptions:
        col = df[inputOption]
        if (col.dtype == int):
            input_dict[inputOption] = st.slider(f"Select hypothetical {inputOption}: [{min(col)}, {max(col)}]", min_value=min(
                col), max_value=max(col))

    prediction = model.predict_model(input_dict)
    st.write(f"""### Probability of `{outputOption}` being true: `{prediction}`""")


main()
