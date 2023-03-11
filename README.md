## Computer vision

This repository has codes developed in a computer vision project where, after implementing several distinct image operations with Python, an **image classification model** was constructed. The dataset used to train and apply this model consists of thousands of pictures of garbage items, so the model has the job of predicting to which of the following classes the picture of an object belongs to: cardboard, glass, metal, paper, plastic, and trash. The complete dataset can be find [here](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification). The hypothetical use of this model implied in a web-application developed using **Streamlit**, and the complete data product that provides the consume of the solution was deployed using **Docker** and **Amazon EC2**.

The complete list of technologies used throughout the project is summarized by the table below:

| **Task**            | **Technology**                                                                                                                                                                                                                                                                |
|---------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Data management     | [pandas](https://pandas.pydata.org/docs/)<br> [numpy](https://numpy.org/doc/)                                                                                                                                                                                                 |
| Image processing    | [Open CV](https://docs.opencv.org/4.x/)<br> [Pillow](https://pillow.readthedocs.io/en/stable/)                                                                                                                                                                                |
| Machine learning    | [Keras](https://keras.io/)<br> [Tensorflow](https://www.tensorflow.org/api_docs)<br> [scikit-learn](https://scikit-learn.org/)<br> [LightGBM](https://lightgbm.readthedocs.io/)                                                                                               |
| Deployment          | [Streamlit](https://docs.streamlit.io/)<br> [Docker](https://www.docker.com/)<br> [DockerHub](https://hub.docker.com/)<br> [Amazon EC2](https://aws.amazon.com/pt/ec2/)<br> [Amazon S3](https://aws.amazon.com/pt/s3/)<br> [Plotly](https://plotly.com/python-api-reference/) |
| Project development | [VS Code](https://code.visualstudio.com/)<br> [Google Colab](https://colab.research.google.com/)                                                                                                                                                                              |

This project has undertaken the traditional [CRISP-DM](https://pt.wikipedia.org/wiki/Cross_Industry_Standard_Process_for_Data_Mining) methodology for the development of the analytical solution constructed here. It is an assumption to the project the absence of Discovery activities (operations understanding, problem definition, solution brainstorm etc.) and the focus on constructing an analytical solution based on an image classification model. Therefore, *only development activities were implemented (i.e., the Delivery phase)*. Such activities are summarized in the following table:

| Stage            | Activities                                                                                                           |
|------------------|----------------------------------------------------------------------------------------------------------------------|
| Data engineering | Reading images<br>Data understanding and cleaning (resize operation).                                                  |
| Data preparation | Data scaling (normalization)<br>Data augmentation (operations of flip, rotation, blur, shift, brightness change, crop) |
| Data modeling    | Architecture and hyper-parameters tuning<br>Model evaluation<br>Model selection<br>Further model analysis          |
| Deployment       | Front-end development<br>Dockerfile and Docker Image construction<br>Application serving                             |
| Documentation    | Description of activities<br>Backlog of future work<br>Conclusions                                                   |

More details regarding development can be found in the *docs* folder and in the [Github page](https://github.com/m-rosso/computer_vision/wiki) of this repository, which consist of the **project documentation**.
