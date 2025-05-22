# 얼굴 이미지 전처리 안내

이 문서는 MTCNN을 사용하여 이미지에서 얼굴을 감지하고 전처리하는 방법을 설명합니다.

## 설치 요구사항

먼저 필요한 라이브러리를 설치해야 합니다:

```bash
pip install facenet-pytorch torch torchvision pillow numpy
```

## 전처리 스크립트 사용법

프로젝트에 포함된 `preprocessing/process_images.py` 스크립트를 사용하여 이미지를 일괄 처리할 수 있습니다.

### 기본 사용법

1. 전처리할 이미지를 `data` 폴더에 저장합니다.
2. 다음 명령어를 실행합니다:

```bash
python preprocessing/process_images.py
```

3. 처리된 이미지는 `processed` 폴더에 저장됩니다.

### 스크립트 동작 방식

1. `data` 폴더에서 모든 이미지 파일(jpg, jpeg, png)을 검색합니다.
2. 각 이미지에서 MTCNN을 사용하여 얼굴을 감지합니다.
3. 감지된 얼굴을 추출하고 전처리합니다:
   - 얼굴 영역에 여백 추가
   - 지정된 크기(기본 160x160)로 조정
   - 픽셀 값 정규화
4. 처리된 얼굴 이미지를 `processed` 폴더에 저장합니다.

### 출력 파일 이름 형식

원본 파일이 `example.jpg`이고 2개의 얼굴이 감지된 경우:
- `processed/example_face1.jpg`
- `processed/example_face2.jpg`

## 코드 커스터마이징

`process_images.py` 스크립트를 직접 수정하여 다음 매개변수를 조정할 수 있습니다:

- `image_size`: 출력 이미지 크기 (기본값: 160)
- `margin`: 얼굴 주변 여백 픽셀 (기본값: 20)
- `min_face_size`: 감지할 최소 얼굴 크기 (기본값: 20)
- `thresholds`: MTCNN 감지 임계값 (기본값: [0.6, 0.7, 0.7])
- `prob_threshold`: 얼굴로 간주할 최소 확률 (기본값: 0.9)

파일 내에서 다음 부분을 수정하여 입력 및 출력 디렉토리를 변경할 수 있습니다:

```python
if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent
    input_dir = os.path.join(base_dir, 'data')        # 입력 디렉토리 변경
    output_dir = os.path.join(base_dir, 'processed')  # 출력 디렉토리 변경
    
    # 매개변수 커스터마이징
    process_images(
        input_dir, 
        output_dir,
        image_size=160,       # 출력 이미지 크기
        margin=20,            # 얼굴 주변 여백
        min_face_size=20,     # 최소 감지 얼굴 크기
        prob_threshold=0.9    # 얼굴 감지 확률 임계값
    )
```

## 문제 해결

- **얼굴이 감지되지 않는 경우**: 이미지의 얼굴이 너무 작거나, 흐릿하거나, 측면을 향하고 있을 수 있습니다. `min_face_size`를 줄이거나 `thresholds` 값을 낮춰보세요.
- **메모리 오류**: 대용량 이미지를 처리할 때 메모리 문제가 발생할 수 있습니다. 이미지를 미리 조정하거나 더 작은 배치로 처리해 보세요.
- **GPU 메모리 부족**: `torch.cuda.empty_cache()`를 주기적으로 호출하거나 CPU에서 처리하세요 (`device='cpu'`).

## 참고 사항

- MTCNN은 프런트 뷰 얼굴 감지에 최적화되어 있습니다. 측면 얼굴은 감지하지 못할 수 있습니다.
- 첫 실행 시 MTCNN 모델 가중치가 자동으로 다운로드됩니다.
- GPU가 있는 경우 처리 속도가 크게 향상됩니다. 