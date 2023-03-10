import streamlit as st
import json

import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), '../../src'
        )
    )
)

with open('config/messages.json', 'r') as json_file:
    messages = json.load(json_file)

st.set_page_config(page_title='ModelMonitor')
st.title('Model monitoring')

st.markdown('#### Future work')
st.write(messages.get('model_monitoring'))
