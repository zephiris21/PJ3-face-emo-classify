# 얼굴 감정 인식 처리기 (Face Emotion Recognition Processor)

이 프로젝트는 facetorch 라이브러리를 사용하여 이미지에서 얼굴을 감지하고 감정을 분석하는 도구입니다.

## 기능
- 이미지에서 얼굴 감지
- 감지된 얼굴의 감정 분석
- 얼굴 정렬 및 저장
- 감정 정보 오버레이된 이미지 생성
- CSV 및 JSON 형식의 결과 생성

## 설치 요구사항
- Python 3.7+
- facetorch 라이브러리
- hydra-core
- PIL
- numpy
- torch
- torchvision

## 설정

### config.json
`config.json` 파일을 통해 주요 설정을 관리할 수 있습니다:

```json
{
  "input": {
    "directory": "./crawling/운명",
    "file_extensions": ["jpg", "jpeg", "png"]
  },
  "output": {
    "base_directory": "./outputs",
    "use_timestamp_folder": true
  },
  "processing": {
    "min_face_size": 96,
    "confidence_threshold": 0.3,
    "batch_size": 1
  },
  "emotion_labels": [
    "Anger", "Contempt", "Disgust", "Fear", 
    "Happiness", "Neutral", "Sadness", "Surprise"
  ]
}
```

설정 옵션:
- `input.directory`: 처리할 이미지가 있는 디렉토리 경로
- `input.file_extensions`: 처리할 이미지 파일 확장자 목록
- `output.base_directory`: 결과 파일이 저장될 기본 디렉토리
- `output.use_timestamp_folder`: 각 실행마다 타임스탬프 기반 하위 폴더 생성 여부
- `processing.min_face_size`: 처리할 최소 얼굴 크기 (픽셀)
- `processing.confidence_threshold`: 학습 데이터로 사용할 최소 신뢰도
- `processing.batch_size`: 배치 처리 크기
- `emotion_labels`: 감정 레이블 목록 (FER 모델의 출력 순서에 맞게 조정 필요)

## 사용 방법
```bash
python face.py
```

## 출력 결과
- `processed_images/`: 감지된 얼굴에 감정 정보가 표시된 이미지
- `aligned_faces/`: 정렬된 얼굴 이미지 (감정 정보 없음)
- `emotion_results.csv`: 감정 분석 결과 (CSV 형식)
- `emotion_results.json`: 감정 분석 결과 (JSON 형식)
- `train_list.txt`: 학습용 이미지 목록

## 참고사항
- facetorch 라이브러리의 설정은 `./facetorch/conf/` 디렉토리에 있습니다.
- unifier 설정이 없는 경우 얼굴 이미지가 생성되지 않을 수 있습니다. 