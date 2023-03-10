FROM python:3.7-slim

WORKDIR /computer_vision/
COPY . /computer_vision/

RUN apt-get update && apt-get install ffmpeg libglib2.0-0 libsm6 libxrender1 libxext6 -y

RUN pip install -r requirements.txt

WORKDIR /computer_vision/app

EXPOSE 8000

CMD ["streamlit", "run", "01_Garbage_classifier.py", "--server.port", "8000"]