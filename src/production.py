###############################################################################################
###############################################################################################
#############################################################LIBRARIES##############################################################

import pandas as pd
import numpy as np
import cv2
from typing import Optional, Union
import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

###############################################################################################
###############################################################################################
########################################################FUNCTIONS AND CLASSES#######################################################

###############################################################################################
# Function that reads and prepares image data as expected by the model developed during experimentation:

def data_prep(path: Optional[str] = None, image: Optional[np.ndarray] = None,
              image_dim: tuple = (224, 224), scale: int = 255) -> tuple:
    """
    Function that reads and prepares image data as expected by the model developed during experimentation.

    :param path: path where the image to be read is located. Should include the file name and extension.
    :type path: string.
    :param image: raw image matrices.
    :type image: numpy nd-arrays.
    :param image_dim: dimension for resizing the image.
    :type image_dim: tuple of integers.
    :param scale: value to divide all pixel values.
    :type scale: integer.

    :return: matrices of original image, resized image, and prepared (resized and scaled) image.
    :rtype: tuple of numpy nd-arrays.
    """
    if image is not None:
        img = cv2.imdecode(image, -1)
    else:
        img = cv2.imread(f'{path}', -1)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized_img = cv2.resize(img, image_dim, interpolation=cv2.INTER_AREA) # Resizing the image.
    scaled_img = resized_img/scale # Scaling the image.
    
    return (img, resized_img, scaled_img)

###############################################################################################
# Function that applies the model over a prepared image and returns predicted probabilities of all classes:

def predict_proba(model, img: np.ndarray) -> np.array:
    """
    Function that applies the model over prepared image and returns predicted probabilities of all classes.

    :param model: trained model for image classification.
    :type model: Tensorflow/Keras object.
    :param img: image prepared as expected by the model developed during experimentation.
    :type img: numpy nd-arrays.

    :return: predicted probabilities that the image belongs to each available class.
    :rtype: numpy array.
    """
    try:
        return model.model.predict(img.reshape((1, *img.shape))).ravel()
    except:
        return model.predict(img.reshape((1, *img.shape))).ravel()

###############################################################################################
# Function that applies the model over a prepared image and returns the predicted class:

def predict(model, img: np.ndarray, label_to_class: dict) -> tuple:
    """
    Function that applies the model over a prepared image and returns the predicted class.

    :param model: trained model for image classification.
    :type model: Tensorflow/Keras object.
    :param img: image prepared as expected by the model developed during experimentation.
    :type img: numpy nd-arrays.
    :param label_to_class: original class of each label (integers 1-6).
    :type label_to_class: dictionary.

    :return: predicted label and class of the image.
    :rtype: tuple of an integer and a string.
    """
    predicted_prob = predict_proba(
        model=model,
        img=img
    ) # Predicted probabilities of each class.
    predicted_label = np.argmax(predicted_prob) + 1
    predicted_class = label_to_class.get(predicted_label)

    return predicted_label, predicted_class

###############################################################################################
# Class to create an outputs table with image identification and predicted probabilities and class:

class Outputs:
    """
    Initialization parameters:
        :param columns: names of columns of outputs table.
        :type columns: list of strings.
        :param dtypes: expected data types of each column.
        :types dtypes: list.
    """
    def __init__(self, columns: list, dtypes: list) -> None:
        self.columns = columns
        self.dtypes = dtypes

        self._table = pd.DataFrame(
            columns=columns
        )
    
    def update(self, data: pd.DataFrame) -> None:
        """
        Method to update the outputs table with provided data.

        :param data: new entry to the table. It should respect the table schema.
        :type data: pandas dataframe.
        """
        self._table = pd.concat(
            [self._table, data],
            axis=0, sort=False
        ).reset_index(drop=True)
        return
    
    def display(self, nrows: Optional[int] = 10, head: bool = True) -> pd.DataFrame:
        """
        Method to display a fraction or the entire outputs table.

        :param nrows: number of rows to be displayed.
        :type nrows: integer.
        :param head: whether the outputs table head (True) or tail (False) should be displayed.
        :type head: boolean.

        :return: fraction or the entire outputs table.
        :rtype: pandas dataframe.
        """
        if nrows is not None:
            if head:
                return self._table.head(nrows)
            else:
                return self._table.tail(nrows)
        else:
            return self._table

###############################################################################################
# Function that initializes an outputs table:

def create_outputs():
    """
    Function that initializes an outputs table.

    :return: instance of Outputs class.
    :rtype: Output class.
    """
    # Schema of outputs table:
    schema = {
        'file_name': str, 'file_type': str,
        'file_size': int, 'epoch': int,
        'cardboard': float,
        'glass': float,
        'metal': float,
        'paper': float,
        'plastic': float,
        'trash': float,
        'pred_class': str, 'true_class': str,
        'model_id': str
    }

    # Creating the outputs table:
    outputs = Outputs(
        columns=list(schema.keys()),
        dtypes=list(schema.values())
    )
    return outputs

###############################################################################################
# Function that creates a new output dataframe:

def create_new_output(image_info: dict, predicted_prob: list, predicted_class: str,
                      app_config: dict) -> Union[dict, pd.DataFrame]:
    """
    Function that creates a new output dataframe.

    :param image_info: information regarding the original input image.
    :type image_info: dictionary.
    :param predicted_prob: predicted probability of each class.
    :type predicted_prob: list.
    :param predicted_class: predicted class.
    :type predicted_class: string.
    :param app_config: app configuration file.
    :type app_config: dictionary.

    :return: new row for the outputs table.
    :rtype: pandas dataframe.
    """
    new_output = {
        'file_name': image_info['file_name'],
        'file_type': image_info['file_type'],
        'file_size': image_info['file_size'],
        'img_shape': str(image_info['img_shape']),
        'epoch': image_info['upload_time'],
        'cardboard': float(predicted_prob[0]),
        'glass': float(predicted_prob[1]),
        'metal': float(predicted_prob[2]),
        'paper': float(predicted_prob[3]),
        'plastic': float(predicted_prob[4]),
        'trash': float(predicted_prob[5]),
        'pred_class': predicted_class,
        'true_class': image_info['image_class'],
        'model_id': app_config['model_id']
    }

    if app_config['storage']!='s3':
        new_output = pd.DataFrame(
            data=new_output,
            index=[0]
        )

    return new_output

###############################################################################################
# Function that returns the graph to be plotted using Streamlit:

def plot_proba(new_output: pd.DataFrame, classes: list, interactive: bool = False) -> None:
    """
    Function that returns the graph to be plotted using Streamlit.

    :param new_output: data with predicted probability by class.
    :type new_output: dataframe.
    :param classes: names of available classes.
    :type classes: list.
    :param interactive: indicates whether to plot interactive or static graph.
    :type interactive: boolean.

    :return: bar plot with predicted probability by class.
    """
    if interactive:
        fig = make_subplots(rows=1, cols=1)

        # Creating the bar plot:
        fig.add_trace(
            go.Bar(
                x=list(new_output[classes].iloc[0].values),
                y=classes,
                name='Predicted probability',
                hovertemplate=f'class = ' + '%{y}<br>probability = %{x:.4f}<br>',
                orientation='h'
            )
        )

        # Customizing the plot:
        fig.update_xaxes(title_text='')
        fig.update_yaxes(title_text='')
        fig.update_traces(marker_color='#4682b4')
        fig.update_layout(
            width=600, height=400,
            showlegend=False,
            title='Predicted probabilities',
            template='simple_white'
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        fig = plt.figure()
        sns.barplot(
            x=new_output[classes].iloc[0].values,
            y=classes,
            color='#4682b4', orient='h'
        )
        plt.title('Predicted probabilities')

        st.pyplot(fig)

###############################################################################################
# Auxiliary functions for a faster front-end development:

def foo_predict_proba():
    random_numbers = np.random.choice(range(1, 100), size=6, replace=True)

    return np.array([i/sum(random_numbers) for i in random_numbers])

def foo_predict(label_to_class: dict) -> tuple:
    random_prob = foo_predict_proba()
    random_label = np.argmax(random_prob) + 1
    random_class = label_to_class.get(random_label)

    return random_label, random_class
