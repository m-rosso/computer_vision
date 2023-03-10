##############################################################################
##############################################################################
###################################LIBRARIES##################################

import pandas as pd
import numpy as np
import streamlit as st
import time
import yaml
from yaml.loader import SafeLoader
import boto3
from dotenv import load_dotenv
import json
from PIL import Image
from io import BytesIO

from keras.models import load_model

import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), '../src'
        )
    )
)

##############################################################################
##############################################################################
###########################FUNCTIONS AND CLASSES##############################

from production import data_prep, create_new_output
from production import foo_predict_proba, foo_predict, plot_proba
from production import predict_proba, predict
from utils import label_to_class

##############################################################################
##############################################################################
###########################DATA AND CONFIGS###################################

# App configuration:
with open('config/app_config.yaml') as yaml_file:
    app_config = yaml.load(yaml_file, Loader=SafeLoader)

# AWS credentials:
load_dotenv()
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

# Connection with AWS API:
client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

# Loading the selected model:
# model = load_model(f'../artifacts/model_{app_config["model_id"]}.h5')

##############################################################################
##############################################################################
##################################APP#########################################

st.set_page_config(page_title='GarbClass')
st.title('Garbage classifier')
st.session_state['displayed_image'] = False

col1, col2 = st.columns(2)

with col1:
    # Uploading input image:
    uploaded_img = st.file_uploader(
        label=app_config['upload_label'],
        help=app_config['upload_help'],
        type=app_config['image_types'],
        accept_multiple_files=False,
        key='image_upload'
    )

with col2:
    if uploaded_img is not None:
        try:
            # Saving information about the input image:
            image_info = {
                'file_name': uploaded_img.name.split('.')[0],
                'file_type': uploaded_img.type,
                'file_size': uploaded_img.size,
                'upload_time': int(time.time())
            }

            # Preparing the input image for the classification model:
            img, resized_img, scaled_img = data_prep(
                image=np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8),
                image_dim=eval(app_config['image_dim_resize']),
                scale=app_config['scale_param']
            )
            image_info['img_shape'] = img.shape

            # Displaying the input image in the app:
            st.image(img, channels='RGB')
            st.session_state['displayed_image'] = True

        except Exception as error:
            st.write(error)

if st.session_state['displayed_image']:
    # Getting the true class of input image:
    image_info['image_class'] = st.selectbox(
        label='Declare the image true class if possible',
        options=['', *app_config['classes']],
        key='true_img_class'
    )

    predict_button = st.button(
        label=app_config['predict_label'],
        key='predict_button',
        help=app_config['predict_help']
    )

    if predict_button:
        with st.spinner('Predicting image class...'):
            # Model predictions for the input image: 
            # predicted_prob = predict_proba(
            #     model=model,
            #     img=scaled_img
            # )
            predicted_prob = foo_predict_proba()
            # predicted_label, predicted_class = predict(
            #     model=model,
            #     img=scaled_img,
            #     label_to_class=label_to_class(classes=app_config['classes'])
            # )
            predicted_label, predicted_class = foo_predict(
                label_to_class(classes=app_config['classes'])
            )

        # Creating new row for the outputs table:
        new_output = create_new_output(
            image_info, predicted_prob, predicted_class, app_config
        )
        
        # Plotting model prediction:
        st.markdown(f'**Predicted class:** {predicted_class}.')
        plot_proba(
            new_output=new_output if app_config['storage']!='s3' else \
                pd.DataFrame(data=new_output, index=[0]),
            classes=app_config['classes'],
            interactive=app_config['interactive_plot']
        )

        # Storing input image information and predictions:
        new_output = json.dumps(new_output, ensure_ascii=False)
        client.put_object(
            Bucket='comp-vision-app',
            Key='inputs_preds/'+image_info['file_name']+'_'+str(image_info['upload_time'])+'.json',
            Body=new_output
        )

        # Storing input image:
        img = Image.fromarray(img)
        out_img = BytesIO()
        img.save(out_img, format='jpeg')
        out_img.seek(0)  
        client.put_object(
            Bucket='comp-vision-app',
            Key='images/'+image_info['file_name']+'_'+str(image_info['upload_time'])+'.jpg',
            Body=out_img,
            ContentType='image/jpg'
        )
