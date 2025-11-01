<h1 align="center"> Sign Language Detection using Machine Learning</h1>

<p align="center">
  <b>Real-time ASL gesture recognition using MediaPipe and LSTM</b><br>
  Recognizes: <b>Hello</b>, <b>Thanks</b>, and <b>I Love You</b>
</p>

<hr>

<h2>➤ Project Overview</h2>
<p>
This project demonstrates a <b>real-time Sign Language Detection System</b> using <b>MediaPipe</b> for keypoint detection and an <b>LSTM</b> neural network for classification.<br>
It captures <b>body, face, and hand landmarks</b> from a webcam and predicts the performed sign gesture.
</p>

<h2>➤ Features</h2>
<ul>
  <li> Real-time sign language recognition from webcam input</li>
  <li> Uses MediaPipe Holistic for hand, face, and pose tracking</li>
  <li> Trains an LSTM-based neural network on gesture sequences</li>
  <li> Evaluates performance using accuracy and confusion matrix</li>
  <li> Saves the trained model for reuse</li>
  <li> Runs efficiently on CPU</li>
</ul>

<hr>

<h2>➤ Tech Stack</h2>
<table>
<tr><th>Component</th><th>Description</th></tr>
<tr><td><b>Language</b></td><td>Python 3.x</td></tr>
<tr><td><b>Frameworks</b></td><td>TensorFlow / Keras</td></tr>
<tr><td><b>Computer Vision</b></td><td>OpenCV, MediaPipe</td></tr>
<tr><td><b>Data Handling</b></td><td>NumPy, scikit-learn, Matplotlib</td></tr>
<tr><td><b>IDE / Platform</b></td><td>Jupyter Notebook</td></tr>
</table>

<hr>

<h2>➤ Project Structure</h2>
<pre>
Sign_Language_Detection_using_Machine_learning/
├── sign.ipynb               # Main Jupyter Notebook
├── MP_Data/                 # Directory for keypoint sequences
│   ├── hello/
│   ├── thanks/
│   └── Iloveyou/
├── model.h5                 # Saved trained LSTM model (optional)
├── README.md                # Project documentation
└── requirements.txt         # Dependencies list (optional)
</pre>

<hr>

<h2>➤ Installation</h2>

<ol>
<li><b>Clone the Repository</b>
<pre>
git clone https://github.com/Ravi6888/Sign_Language_Detection_using_Machine_learning.git
cd Sign_Language_Detection_using_Machine_learning
</pre></li>

<li><b>Install Dependencies</b>
<pre>
pip install opencv-python mediapipe numpy tensorflow matplotlib scikit-learn
</pre></li>

<li><b>Launch the Notebook</b>
<pre>
jupyter notebook sign.ipynb
</pre></li>
</ol>

<hr>

<h2>➤ Dataset & Keypoint Collection</h2>
<ul>
  <li>Data is automatically collected using your webcam.</li>
  <li>Each gesture (class) consists of 30 sequences, each containing 30 frames.</li>
  <li>Frames are stored as <code>.npy</code> files with MediaPipe keypoints (pose + face + hands).</li>
</ul>

<pre>
MP_Data/
├── hello/
│   ├── 0/ → frame_1.npy, frame_2.npy, ...
│   ├── 1/
│   └── ...
├── thanks/
└── Iloveyou/
</pre>

<hr>

<h2>➤ Model Architecture</h2>
<pre>
Model: Sequential
──────────────────────────────────────
1. LSTM Layer (64 units)
2. LSTM Layer (128 units)
3. Dense Layer (64 units, ReLU)
4. Dense Layer (32 units, ReLU)
5. Output Layer (Softmax, 3 classes)
──────────────────────────────────────
Optimizer: Adam
Loss Function: Categorical Crossentropy
Metrics: Accuracy
</pre>

<hr>

<h2>➤ Model Training</h2>
<ul>
  <li>Trains on labeled keypoint sequences from webcam input.</li>
  <li>Uses <code>train_test_split</code> for validation.</li>
  <li>Monitored with TensorBoard for training visualization.</li>
  <li>Achieves high accuracy for three defined gestures.</li>
</ul>

<hr>

<h2>➤ Evaluation</h2>
<ul>
  <li>Evaluated with confusion matrix and accuracy score (scikit-learn).</li>
  <li>Performs reliable real-time predictions for the gestures.</li>
</ul>

<hr>

<h2>➤ Real-Time Testing</h2>
<p>
Run the final cell in the notebook for live testing:<br>
• Opens a webcam window<br>
• Tracks hand and pose landmarks in real-time<br>
• Displays the predicted sign on the video feed<br>
Press <b>q</b> to quit the webcam stream.
</p>

<pre>
[INFO] Starting webcam...
Detected: Hello
Detected: Thanks
Detected: I Love You
</pre>

<hr>

<h2>➤ Possible Enhancements</h2>
<ul>
  <li>Add more gestures and languages (Indian / ASL / BSL)</li>
  <li>Include full-body pose cues for context-rich gestures</li>
  <li>Add text-to-speech conversion for recognized signs</li>
  <li>Deploy using Streamlit or Flask for a web interface</li>
  <li>Convert model to TensorFlow Lite for mobile apps</li>
</ul>

<hr>



<hr>

<p align="center">
⭐ <i>If you found this project useful, consider giving it a star!</i> ⭐
</p>
