# 얼굴 인식 및 이미지 전처리 가이드 - MTCNN 사용법

이 문서는 `facenet_pytorch` 라이브러리의 `MTCNN` 모듈을 사용하여 이미지에서 얼굴을 감지하고 전처리하는 방법을 설명합니다.

## 필요한 라이브러리 설치

```bash
pip install facenet-pytorch
pip install torch torchvision
pip install pillow numpy
```

## 기본 사용법

### 1. 모듈 불러오기

```python
from facenet_pytorch import MTCNN
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
```

### 2. MTCNN 모델 초기화

```python
# GPU 사용 가능 여부 확인
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# MTCNN 모델 초기화
# image_size: 출력 이미지 크기
# margin: 얼굴 주변에 추가할 여백(픽셀)
mtcnn = MTCNN(
    image_size=160,  # 출력 이미지 크기 (정사각형)
    margin=20,       # 얼굴 주변 여백
    min_face_size=20,  # 감지할 최소 얼굴 크기
    thresholds=[0.6, 0.7, 0.7],  # 각 단계별 감지 임계값
    factor=0.709,    # 이미지 피라미드 스케일 인자
    post_process=True,  # 후처리 여부
    device=device    # 연산 장치 (CPU/GPU)
)
```

### 3. 이미지 로드

다양한 형식의 이미지를 처리할 수 있습니다:

```python
# 방법 1: PIL Image 객체로 로드
img = Image.open('sample.jpg')

# 방법 2: numpy 배열로 로드 (RGB 형식, shape: [H, W, 3])
import cv2
img_cv = cv2.imread('sample.jpg')
img_cv_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)  # OpenCV는 BGR 형식으로 로드하므로 RGB로 변환
```

### 4. 얼굴 감지 및 크롭

#### 4.1 단일 이미지에서 얼굴 감지

```python
# 얼굴 위치 및 확률 감지
boxes, probs = mtcnn.detect(img)

# 감지된 얼굴이 있는 경우
if boxes is not None:
    # 각 얼굴에 대해 처리
    for i, (box, prob) in enumerate(zip(boxes, probs)):
        # 일정 확률 이상인 경우만 처리
        if prob > 0.9:
            # 박스 좌표 (x1, y1, x2, y2)
            x1, y1, x2, y2 = box.astype(int)
            
            # 얼굴 부분 크롭
            face = img.crop((x1, y1, x2, y2))
            
            # 필요한 경우 크기 조정
            face_resized = face.resize((160, 160))
            
            # 결과 저장 또는 처리
            face_resized.save(f'face_{i}.jpg')
```

#### 4.2 MTCNN의 내장 함수 사용

```python
# 얼굴 크롭 및 전처리를 한번에 진행 (정사각형 이미지로 변환)
face_img = mtcnn(img)  # 반환값: 전처리된 tensor (RGB 채널 이미지)

# 여러 얼굴 처리
batch_boxes, batch_probs, batch_points = mtcnn.detect(img, landmarks=True)
faces = mtcnn.extract(img, batch_boxes, save_path=None)
```

### 5. 결과 시각화

```python
def visualize_detection(img, boxes, probs):
    """감지된 얼굴에 사각형 표시"""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(img)
    
    # 감지된 얼굴이 있는 경우
    if boxes is not None:
        for i, (box, prob) in enumerate(zip(boxes, probs)):
            if prob > 0.9:  # 확률이 높은 얼굴만 표시
                x1, y1, x2, y2 = box.astype(int)
                
                # 사각형 그리기
                rect = patches.Rectangle(
                    (x1, y1), x2-x1, y2-y1, 
                    linewidth=2, edgecolor='r', facecolor='none'
                )
                ax.add_patch(rect)
                ax.text(x1, y1-10, f'Face {i+1}: {prob:.2f}', color='red')
    
    plt.axis('off')
    plt.show()

# 시각화 함수 호출
if boxes is not None:
    visualize_detection(img, boxes, probs)
```

## 입력 이미지 요구사항

- **형식**: RGB 이미지 (PIL Image 객체 또는 numpy 배열 [H, W, 3])
- **크기**: 특별한 제한은 없으나, 너무 크거나 작은 이미지는 성능에 영향을 줄 수 있음
- **전처리**: 특별한 전처리가 필요 없음. 원본 이미지를 바로 사용 가능
- **얼굴 크기**: `min_face_size` 매개변수보다 큰 얼굴만 감지
- **이미지 품질**: 명확한 얼굴이 있는 이미지에서 최상의 결과

## 이미지 크기 조정 및 정규화

MTCNN은 다음과 같은 전처리를 자동으로 수행합니다:

1. **검출**: 다양한 크기의 얼굴을 감지
2. **크롭**: 감지된 얼굴 영역을 추출
3. **마진 추가**: 설정된 마진에 따라 얼굴 주변에 여백 추가
4. **리사이징**: `image_size` 매개변수에 따라 일정한 크기로 조정 (기본값: 160x160)
5. **정규화**: 픽셀 값 범위를 [0, 1]로 정규화

## 전체 예제 코드

```python
from facenet_pytorch import MTCNN
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 장치 설정
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Running on device: {device}')

# MTCNN 모델 초기화
mtcnn = MTCNN(
    image_size=160,
    margin=20,
    min_face_size=20,
    thresholds=[0.6, 0.7, 0.7],
    factor=0.709,
    post_process=True,
    device=device
)

# 이미지 로드
img_path = 'sample.jpg'
img = Image.open(img_path)

# 얼굴 감지
boxes, probs = mtcnn.detect(img)

# 결과 시각화
def visualize_detection(img, boxes, probs):
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(img)
    
    if boxes is not None:
        for i, (box, prob) in enumerate(zip(boxes, probs)):
            if prob > 0.9:
                x1, y1, x2, y2 = box.astype(int)
                
                rect = patches.Rectangle(
                    (x1, y1), x2-x1, y2-y1, 
                    linewidth=2, edgecolor='r', facecolor='none'
                )
                ax.add_patch(rect)
                ax.text(x1, y1-10, f'Face {i+1}: {prob:.2f}', color='red')
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('detected_faces.jpg', bbox_inches='tight')
    plt.show()

# 얼굴 감지 및 저장
if boxes is not None:
    visualize_detection(img, boxes, probs)
    
    # 각 얼굴 크롭 및 저장
    for i, (box, prob) in enumerate(zip(boxes, probs)):
        if prob > 0.9:
            # 전처리된 얼굴 추출
            face = mtcnn.extract(img, [box], save_path=None)[0]
            
            # Tensor를 PIL 이미지로 변환
            if face is not None:
                face_img = Image.fromarray((face.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
                face_img.save(f'face_{i+1}.jpg')
                print(f'얼굴 {i+1} 저장 완료, 확률: {prob:.2f}')
else:
    print("얼굴이 감지되지 않았습니다.")
```

## 참고사항

- MTCNN은 Multi-task Cascaded Convolutional Networks의 약자로, 다중 단계(cascade)를 통해 얼굴을 감지합니다.
- 처음 사용 시 자동으로 가중치 파일을 다운로드합니다.
- 큰 이미지에서 여러 얼굴을 감지할 때는 GPU 사용을 권장합니다.
- GPU를 사용할 경우 `torch.cuda.empty_cache()`를 통해 주기적으로 메모리를 정리하는 것이 좋습니다. 