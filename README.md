# Sign_Language_Detection_using_Machine_learning

	**Project Overview**

This project demonstrates a real-time Sign Language Detection System using MediaPipe for keypoint detection and a Long Short-Term Memory (LSTM) neural network for classification.

The system captures body, face, and hand landmarks from a webcam and predicts the performed sign gesture.

It currently recognizes three American Sign Language (ASL) gestures:

 Hello, Thanks, and I Love You

________________________________________

	**Features**

•	 Real-time sign language recognition from webcam input

•	 Uses MediaPipe Holistic for hand, face, and pose tracking

•	 Trains an LSTM-based neural network on gesture sequences

•	 Evaluates performance using accuracy and confusion matrix

•	 Saves the trained model for reuse

•	 Runs efficiently on CPU

________________________________________

	**Tech Stack**

Component	&nbsp;&nbsp;&nbsp;&nbsp; Description

Language &nbsp;&nbsp;&nbsp;&nbsp;	Python 3.x

Frameworks &nbsp;&nbsp;&nbsp;&nbsp;	TensorFlow / Keras

Computer Vision	&nbsp;&nbsp;&nbsp;&nbsp;OpenCV, MediaPipe

Data Handling	&nbsp;&nbsp;&nbsp;&nbsp; NumPy, scikit-learn, Matplotlib

IDE/Platform &nbsp;&nbsp;&nbsp;&nbsp;	Jupyter Notebook

________________________________________

	**Project Structure**

Sign_Language_Detection_using_Machine_learning/

│

|── sign.ipynb            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   # Main Jupyter Notebook

|── MP_Data/         &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;        # Directory for keypoint sequences

│   |── hello/

│   |── thanks/

│   └── Iloveyou/
 

________________________________________

	**Installation**



1.	Clone the Repository 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; git clone https://github.com/Ravi6888/Sign_Language_Detection_using_Machine_learning.git 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; cd Sign_Language_Detection_using_Machine_learning

2.	 Install Dependencies 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; pip install opencv-python mediapipe numpy tensorflow matplotlib scikit-learn

3.	 Launch the Notebook 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; jupyter notebook sign.ipynb

________________________________________

	**Dataset & Keypoint Collection**

•	Data is automatically collected using your webcam.

•	Each gesture (class) consists of 30 sequences, and each sequence contains 30 frames.

•	The frames are stored in the following structure:

MP_Data/

 |── hello/
 
 │   |── 0/ → frame_1.npy, frame_2.npy, ...
 
 │   |── 1/
 
 │   └── ...
 
 |── thanks/
 
 └── Iloveyou/

•Each .npy file stores the MediaPipe keypoints (pose + face + hands).

________________________________________

	**Model Architecture**

The project uses a Sequential LSTM Network trained on keypoint sequences.

Model: Sequential

──────────────────────────────────────

1. LSTM Layer (64 units)

2. LSTM Layer (128 units)

3. Dense Layer (64 units, ReLU)

4. Dense Layer (32 units, ReLU)

5. Output Layer (Softmax, 3 classes)

──────────────────────────────────────

•	Optimizer: Adam

•	Loss Function: Categorical Crossentropy

•	Metrics: Accuracy

________________________________________

	**Model Training**

•	Trains the model using the labeled sequences generated from webcam input.

•	Uses train_test_split for validation.

•	Employs TensorBoard for visualizing training progress.

•	Achieves high accuracy for the three gestures.

________________________________________

	**Evaluation**

•	Evaluated using:

•	Confusion Matrix

•	Accuracy Score from scikit-learn

•	The model achieves reliable real-time predictions for the defined gestures.

________________________________________

	**Real-Time Testing**

Run the final cell in the notebook for live testing:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; •	Opens a webcam window

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; •	Tracks hand and pose landmarks in real-time

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; •	Displays the predicted sign on the video feed

Press q to quit the webcam stream.

________________________________________

	**Example Output**

[INFO] Starting webcam...

Detected: Hello

Detected: Thanks

Detected: I Love You ️

________________________________________

	**Possible Enhancements**

•	 Add more gestures and languages (Indian / ASL / BSL).

•	 Include full-body pose cues for context-rich gestures.

•	 Add text-to-speech conversion of recognized signs.

•	 Deploy using Streamlit or Flask for a web interface.

•	 Convert model to TensorFlow Lite for mobile apps.

