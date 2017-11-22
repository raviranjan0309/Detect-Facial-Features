# Detect-Facial-Features

This tutorial will help you to extract the cordinates for facial features like eyes, nose, mouth and jaw using 68 facial landmark indexes.

### 68 Facial landmark indexes

The facial landmark detector implemented inside dlib produces 68 (x, y)-coordinates that map to specific facial structures. These 68 point mappings were obtained by training a shape predictor on the labeled iBUG 300-W dataset.

Below we can visualize what each of these 68 coordinates map to:

![N|Solid](https://www.pyimagesearch.com/wp-content/uploads/2017/04/facial_landmarks_68markup.jpg)

Examining the image, we can see that facial regions can be accessed via simple Python indexing (assuming zero-indexing with Python since the image above is one-indexed):

 - The mouth can be accessed through points [48, 68].
 - The right eyebrow through points [17, 22].
 - The left eyebrow through points [22, 27].
 - The right eye using [36, 42].
 - The left eye with [42, 48].
 - The nose using [27, 35].
 - And the jaw via [0, 17].
 
These mappings are encoded inside the FACIAL_LANDMARKS_IDXS  dictionary inside face_utils of the imutils library.

### Tech

Project uses below python packages:

* [NumPy](http://www.numpy.org/) - A fundamental package for scientific computing with Python.
* [OpenCV](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_setup/py_intro/py_intro.html) - A library of Python bindings designed to solve computer vision problems.
* [dlib](https://pypi.python.org/pypi/dlib) - A toolkit for making real world machine learning and data analysis applications.
* [imutils](https://pypi.python.org/pypi/imutils) - A series of convenience functions to make basic image processing functions such as translation, rotation, resizing, skeletonization, displaying Matplotlib images, sorting contours, detecting edges, and much more easier with OpenCV and both Python 2.7 and Python 3.


### Installation Steps for running on your local machine:

 - Download and install Python 3.6 or Anaconda 4 on your system as per your operating system [Download Python](https://www.python.org/downloads/release/python-360/) [Download Anaconda 4](https://www.anaconda.com/download/)
 - Clone the repository on your local drive.
 - Open command prompt in windows or terminal in MacOS inside the application root directory.
 - Install the required python packages needed to run the appliation.
     ```sh
     pip install requirements.txt
     ```
 - Run below command to visualize the results
    ```sh
     python detect_face_features.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/image_1.jpg
    ```

##### Author    
##### Ravi Ranjan














