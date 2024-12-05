Violence detection by CNN model

In order to identify violent behaviors in multimedia content, the pre-trained ResNet50 convolutional neural networks image recognition architecture was used. The resulting model was used to feed a web application for image and video recognition. In the case of video, a number of frames are captured to generalize a classification based on the number of frames belonging to both classes.

In ordertodeploylocally the app, just got to root directory and write in the CLI:
streamlit run app.py