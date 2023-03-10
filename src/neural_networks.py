####################################################################################################################################
####################################################################################################################################
#############################################################LIBRARIES##############################################################

import pandas as pd
import numpy as np
from typing import Union

from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import Callback

####################################################################################################################################
####################################################################################################################################
########################################################FUNCTIONS AND CLASSES#######################################################

####################################################################################################################################
# Class for evaluating a neural network at each epoch based on validation data:

class AccuracyEpochs(Callback):
    """
    Class for evaluating a neural network at each epoch based on validation data.
    Besides, this class implements early stopping based on a maximum number of
    epochs without increasing in validation accuracy.

    :param val_inputs: inputs for validation data.
    :type val_inputs: dataframe or matrix.
    :param val_output: outputs for validation data.
    :type val_output: iterable.
    :param early_stopping: indicates whether early stopping should take place.
    :type early_stopping: boolean.
    :param patience: number of epochs without increasing in validation accuracy.
    :type patience: integer.
    :param min_increase: defines minimum necessary increasing.
    :type min_increase: float.
    """
    def __init__(self, val_inputs: Union[pd.DataFrame, np.array],
                 val_output: Union[list, np.array, pd.Series],
                 early_stopping: bool = False, patience: int = 10, min_increase: float = 0):
        super(Callback, self).__init__()
        self.val_inputs = val_inputs
        self.val_output = val_output
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_increase = min_increase
        self.val_accuracy = []

        if self.early_stopping:
            self.last_acc = 0
            self.check = 0

    # Assessment at the end of epoch:
    def on_epoch_end(self, batch, logs={}) -> None:
        # Accuracy at the current epoch:
        current_accuracy = accuracy_score(
            self.val_output,
            [np.argmax(p) for p in self.model.predict(self.val_inputs)]
        )

        # Calculating accuracy at the end of epoch:
        self.val_accuracy.append(current_accuracy)

        # Checking early stopping criterium:
        if self.early_stopping:
            if current_accuracy <= self.last_acc + self.min_increase:
                self.check += 1
            else:
                self.best_model = self.model
            
            self.last_acc = current_accuracy

            if self.check == self.patience:
                self.model.stop_training = True
