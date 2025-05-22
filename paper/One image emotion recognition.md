# Recognize emotions on one image with Python interface of EmotiEffLib

The first GPU device should be used for cuda


```python
%env CUDA_VISIBLE_DEVICES=0
```

    env: CUDA_VISIBLE_DEVICES=0
    

Function for faces recognition:


```python
from typing import List
import numpy as np

def recognize_faces(frame: np.ndarray, device: str) -> List[np.array]:
    """
    Detects faces in the given image and returns the facial images cropped from the original.

    This function reads an image from the specified path, detects faces using the MTCNN
    face detection model, and returns a list of cropped face images.

    Args:
        frame (numpy.ndarray): The image frame in which faces need to be detected.
        device (str): The device to run the MTCNN face detection model on, e.g., 'cpu' or 'cuda'.

    Returns:
        list: A list of numpy arrays, representing a cropped face image from the original image.

    Example:
        faces = recognize_faces('image.jpg', 'cuda')
        # faces contains the cropped face images detected in 'image.jpg'.
    """

    def detect_face(frame: np.ndarray):
        mtcnn = MTCNN(keep_all=False, post_process=False, min_face_size=40, device=device)
        bounding_boxes, probs = mtcnn.detect(frame, landmarks=False)
        if probs[0] is None:
            return []
        bounding_boxes = bounding_boxes[probs > 0.9]
        return bounding_boxes

    bounding_boxes = detect_face(frame)
    facial_images = []
    for bbox in bounding_boxes:
        box = bbox.astype(int)
        x1, y1, x2, y2 = box[0:4]
        facial_images.append(frame[y1:y2, x1:x2, :])
    return facial_images
```

Check if it runs under colab and install dependencies:


```python
try:
    import google.colab
    import urllib.request
    IN_COLAB = True
    urllib.request.urlretrieve("https://github.com/sb-ai-lab/EmotiEffLib/blob/main/docs/tutorials/python/requirements.txt?raw=true", "requirements.txt")
    !pip install -r requirements.txt
except:
    IN_COLAB = False
```

Helper function to get path to the input image:


```python
import urllib.request

def get_test_image(test_dir):
    input_file = os.path.join(test_dir, "test_images", "20180720_174416.jpg")
    if os.path.isfile(input_file):
        return input_file
    url = "https://github.com/sb-ai-lab/EmotiEffLib/blob/main/tests/test_images/20180720_174416.jpg?raw=true"
    print("Downloading input image from", url)
    input_file = "20180720_174416.jpg"
    if os.path.isfile(input_file):
        return input_file
    urllib.request.urlretrieve(url, input_file)
    return input_file
```

## EmotiEffLib with ONNX

Install EmotiEffLib with ONNX support:


```python
!pip install emotiefflib
```

    Requirement already satisfied: emotiefflib in /opt/anaconda3/envs/jupyter/lib/python3.11/site-packages (0.4.0)
    Requirement already satisfied: numpy in /opt/anaconda3/envs/jupyter/lib/python3.11/site-packages (from emotiefflib) (1.26.4)
    Requirement already satisfied: onnx in /opt/anaconda3/envs/jupyter/lib/python3.11/site-packages (from emotiefflib) (1.17.0)
    Requirement already satisfied: onnxruntime in /opt/anaconda3/envs/jupyter/lib/python3.11/site-packages (from emotiefflib) (1.20.1)
    Requirement already satisfied: opencv-python in /opt/anaconda3/envs/jupyter/lib/python3.11/site-packages (from emotiefflib) (4.11.0.86)
    Requirement already satisfied: pillow in /opt/anaconda3/envs/jupyter/lib/python3.11/site-packages (from emotiefflib) (10.2.0)
    Requirement already satisfied: protobuf>=3.20.2 in /opt/anaconda3/envs/jupyter/lib/python3.11/site-packages (from onnx->emotiefflib) (3.20.3)
    Requirement already satisfied: coloredlogs in /opt/anaconda3/envs/jupyter/lib/python3.11/site-packages (from onnxruntime->emotiefflib) (15.0.1)
    Requirement already satisfied: flatbuffers in /opt/anaconda3/envs/jupyter/lib/python3.11/site-packages (from onnxruntime->emotiefflib) (25.2.10)
    Requirement already satisfied: packaging in /opt/anaconda3/envs/jupyter/lib/python3.11/site-packages (from onnxruntime->emotiefflib) (24.2)
    Requirement already satisfied: sympy in /opt/anaconda3/envs/jupyter/lib/python3.11/site-packages (from onnxruntime->emotiefflib) (1.13.3)
    Requirement already satisfied: humanfriendly>=9.1 in /opt/anaconda3/envs/jupyter/lib/python3.11/site-packages (from coloredlogs->onnxruntime->emotiefflib) (10.0)
    Requirement already satisfied: mpmath<1.4,>=1.1.0 in /opt/anaconda3/envs/jupyter/lib/python3.11/site-packages (from sympy->onnxruntime->emotiefflib) (1.3.0)
    

Import libraries:


```python
import os
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
from facenet_pytorch import MTCNN
from emotiefflib.facial_analysis import EmotiEffLibRecognizer, get_model_list
```

    2025-02-27 19:33:11.273060: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    

Path to the test directory, select device and model:


```python
test_dir = os.path.join("..", "..", "..", "tests")
device = "cpu"
model_name = get_model_list()[0]
```

Open test image:


```python
input_file = get_test_image(test_dir)
frame_bgr = cv2.imread(input_file)
frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(5, 5))
plt.axis('off')
plt.imshow(frame)
```




    <matplotlib.image.AxesImage at 0x14e6b9710>




    
![png](One%20image%20emotion%20recognition_files/One%20image%20emotion%20recognition_17_1.png)
    


### Recognize faces on the test image and detect emotions:


```python
facial_images = recognize_faces(frame, device)

fer = EmotiEffLibRecognizer(engine="onnx", model_name=model_name, device=device)

emotions = []
for face_img in facial_images:
    emotion, _ = fer.predict_emotions(face_img, logits=True)
    emotions.append(emotion[0])
    plt.figure(figsize=(3, 3))
    plt.axis('off')
    plt.imshow(face_img)
    plt.title(emotion[0])
```


    
![png](One%20image%20emotion%20recognition_files/One%20image%20emotion%20recognition_19_0.png)
    



    
![png](One%20image%20emotion%20recognition_files/One%20image%20emotion%20recognition_19_1.png)
    



    
![png](One%20image%20emotion%20recognition_files/One%20image%20emotion%20recognition_19_2.png)
    


### Detect emotions on several facial images:


```python
multi_emotions, _ = fer.predict_emotions(facial_images, logits=True)
assert multi_emotions == emotions
```

### Detect emotions by calling features extractor and classifier separately


```python
features = fer.extract_features(facial_images)
classified_emotions, _ = fer.classify_emotions(features, logits=True)
assert classified_emotions == emotions
```

## EmotiEffLib with PyTorch

Install EmotiEffLib with ONNX and Torch support:


```python
!pip install emotiefflib[torch]
```

    zsh:1: no matches found: emotiefflib[torch]
    

Import libraries:


```python
import os
from typing import List

import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
from facenet_pytorch import MTCNN
from emotiefflib.facial_analysis import EmotiEffLibRecognizer, get_model_list
```

Path to the test directory, select device and model:


```python
test_dir = os.path.join("..", "..", "..", "tests")
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = get_model_list()[0]
```

Open test image:


```python
input_file = get_test_image(test_dir)
frame_bgr = cv2.imread(input_file)
frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(5, 5))
plt.axis('off')
plt.imshow(frame)
```




    <matplotlib.image.AxesImage at 0x15028cf10>




    
![png](One%20image%20emotion%20recognition_files/One%20image%20emotion%20recognition_32_1.png)
    


### Recognize faces on the test image and detect emotions:


```python
facial_images = recognize_faces(frame, device)

fer = EmotiEffLibRecognizer(engine="torch", model_name=model_name, device=device)

emotions = []
for face_img in facial_images:
    emotion, _ = fer.predict_emotions(face_img, logits=True)
    emotions.append(emotion[0])
    plt.figure(figsize=(3, 3))
    plt.axis('off')
    plt.imshow(face_img)
    plt.title(emotion[0])
```


    
![png](One%20image%20emotion%20recognition_files/One%20image%20emotion%20recognition_34_0.png)
    



    
![png](One%20image%20emotion%20recognition_files/One%20image%20emotion%20recognition_34_1.png)
    



    
![png](One%20image%20emotion%20recognition_files/One%20image%20emotion%20recognition_34_2.png)
    


### Detect emotions on several facial images:


```python
multi_emotions, _ = fer.predict_emotions(facial_images, logits=True)
assert multi_emotions == emotions
```

### Detect emotions by calling features extractor and classifier separately


```python
features = fer.extract_features(facial_images)
classified_emotions, _ = fer.classify_emotions(features, logits=True)
assert classified_emotions == emotions
```
