import os
import glob
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN

def process_images(input_dir, output_dir, image_size=160, margin=20, min_face_size=20, 
                  thresholds=[0.6, 0.7, 0.7], prob_threshold=0.9):
    """
    이미지 디렉토리에서 얼굴을 감지하고 전처리하여 출력 디렉토리에 저장합니다.
    
    Args:
        input_dir (str): 입력 이미지 디렉토리 경로
        output_dir (str): 출력 이미지 디렉토리 경로
        image_size (int): 출력 이미지 크기
        margin (int): 얼굴 주변 여백 (픽셀)
        min_face_size (int): 감지할 최소 얼굴 크기
        thresholds (list): MTCNN 단계별 임계값
        prob_threshold (float): 얼굴 감지 확률 임계값
    """
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 장치 설정
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Running on device: {device}')
    
    # MTCNN 모델 초기화
    mtcnn = MTCNN(
        image_size=image_size,
        margin=margin,
        min_face_size=min_face_size,
        thresholds=thresholds,
        factor=0.709,
        post_process=True,
        device=device
    )
    
    # 이미지 파일 목록 가져오기 (jpg, jpeg, png)
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    
    print(f'Found {len(image_paths)} images in {input_dir}')
    
    # 각 이미지 처리
    for idx, img_path in enumerate(image_paths):
        try:
            # 이미지 로드
            img = Image.open(img_path)
            img_name = os.path.basename(img_path)
            img_name_base, img_ext = os.path.splitext(img_name)
            
            print(f'Processing image {idx+1}/{len(image_paths)}: {img_name}')
            
            # 얼굴 감지
            boxes, probs = mtcnn.detect(img)
            
            if boxes is not None:
                # 각 얼굴 처리
                for i, (box, prob) in enumerate(zip(boxes, probs)):
                    if prob > prob_threshold:
                        print(f'  Face {i+1} detected with probability: {prob:.4f}')
                        
                        # 방법 1: 직접 크롭 및 리사이징 수행
                        x1, y1, x2, y2 = [int(b) for b in box]
                        
                        # 마진 추가 (이미지 경계 확인)
                        x1 = max(0, x1 - margin)
                        y1 = max(0, y1 - margin)
                        x2 = min(img.width, x2 + margin)
                        y2 = min(img.height, y2 + margin)
                        
                        # 얼굴 영역 크롭
                        face_img = img.crop((x1, y1, x2, y2))
                        
                        # 크기 조정
                        face_img = face_img.resize((image_size, image_size), Image.BILINEAR)
                        
                        # 출력 파일 경로
                        output_path = os.path.join(
                            output_dir, 
                            f"{img_name_base}_face{i+1}{img_ext}"
                        )
                        
                        # 이미지 저장
                        face_img.save(output_path)
                        print(f'  Saved to {output_path}')
                    else:
                        print(f'  Face {i+1} detected but below threshold: {prob:.4f}')
            else:
                print(f'  No faces detected in {img_name}')
                
        except Exception as e:
            print(f'Error processing {img_path}: {str(e)}')
    
    print('Processing complete!')

if __name__ == "__main__":
    # 현재 디렉토리 기준으로 상대 경로 설정
    base_dir = Path(__file__).parent.parent
    input_dir = os.path.join(base_dir, 'crawling/부당거래')
    output_dir = os.path.join(base_dir, 'processed')
    
    # 이미지 처리 실행
    process_images(input_dir, output_dir)