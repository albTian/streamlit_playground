import random
from typing import Dict, List

import pandas as pd
from sklearn import ensemble, linear_model, neural_network

MODEL_TYPES = [
    linear_model.LogisticRegression,
    neural_network.MLPClassifier,
    ensemble.RandomForestClassifier,
]

# TODO: abstract this into a class so we keep the model object under the hood


class SimpleModel:
    model: any
    modelOutput: str
    modelInputs: List[str]

    def __init__(self, _modelOutput, _modelInputs) -> None:
        self.modelOutput = _modelOutput
        self.modelInputs = _modelInputs

    def train_model(
            self,
            dataset: pd.DataFrame,) -> any:
        # self.modelInputs = inputs
        # self.modelOutput = output
        assert dataset[self.modelOutput].dtype in (bool,)
        assert all(dataset[input].dtype in (float, int) for input in self.modelInputs)

        X = dataset[self.modelInputs]
        y = dataset[self.modelOutput]

        self.model = random.choice(MODEL_TYPES)()
        self.model.fit(X, y)

    def predict_model(self, hypothetical_input: Dict[str, float]):
        return self.model.predict_proba(
            pd.DataFrame({input: [hypothetical_input[input]]
                         for input in self.modelInputs})
        )[0, self.model.classes_.tolist().index(True)]
