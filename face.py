import os
import glob
import csv
import json
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
# facetorch 모듈을 직접 로컬 경로에서 가져오기
import sys
sys.path.append(os.path.abspath('.'))  # 현재 디렉토리를 Python 경로에 추가
try:
    # facetorch가 이미 설치되어 있으므로 패키지에서 직접 import
    from facetorch.analyzer import FaceAnalyzer
except ImportError as e:
    print(f"facetorch 모듈을 찾을 수 없습니다: {e}")
    print("facetorch 패키지가 정상적으로 설치되었는지 확인하세요.")
    sys.exit(1)
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
import shutil
from torchvision import transforms
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def softmax(logits):
    """
    로짓에 softmax를 적용하여 확률 값으로 변환
    """
    if isinstance(logits, torch.Tensor):
        return F.softmax(logits, dim=0)
    else:
        # 로짓이 numpy 배열이나 리스트인 경우
        logits_tensor = torch.tensor(logits)
        return F.softmax(logits_tensor, dim=0).numpy()

def load_config():
    """
    config.json 파일에서 설정을 로드하고 기본값과 병합
    """
    config_path = "config.json"
    
    # 기본 설정 값
    default_config = {
        "input": {
            "directory": "./crawling/운명",
            "file_extensions": ["jpg", "jpeg", "png"]
        },
        "output": {
            "base_directory": "./outputs",
            "use_timestamp_folder": True
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
    
    # config.json 파일이 존재하는 경우 로드
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
            
            # 기본 설정과 사용자 설정 병합
            # 딥 병합 (nested dictionaries)
            def deep_update(source, updates):
                for key, value in updates.items():
                    if key in source and isinstance(source[key], dict) and isinstance(value, dict):
                        deep_update(source[key], value)
                    else:
                        source[key] = value
                return source
            
            config = deep_update(default_config.copy(), user_config)
            logger.info(f"설정 파일 로드 완료: {config_path}")
        except Exception as e:
            logger.error(f"설정 파일 로드 오류: {str(e)}")
            logger.info("기본 설정 사용")
            config = default_config
    else:
        logger.warning(f"설정 파일이 없습니다: {config_path}")
        logger.info("기본 설정 사용")
        config = default_config
    
    return config

def extract_emotion(face, emotion_labels):
    """
    얼굴 객체에서 감정 정보 추출
    """
    emotion = "Unknown"
    confidence = 0.0
    all_scores = {}
    
    # 감정 예측 결과 추출
    if "fer" in face.preds:
        fer_pred = face.preds["fer"]
        
        # 레이블 정보 확인
        if hasattr(fer_pred, "label") and fer_pred.label:
            emotion = fer_pred.label
        
        # 로짓에서 confidence 계산
        if hasattr(fer_pred, "logits") and fer_pred.logits.numel() > 0:
            # softmax 적용하여 확률 값으로 변환
            probs = softmax(fer_pred.logits)
            # 최대 확률 및 해당 인덱스 추출
            max_prob, max_idx = torch.max(probs, dim=0)
            confidence = float(max_prob.item())
            
            # 레이블이 없는 경우 인덱스로 감정 추출
            if emotion == "Unknown" and 0 <= max_idx < len(emotion_labels):
                emotion = emotion_labels[max_idx]
            
            # 모든 감정 점수 저장
            for j, score in enumerate(probs):
                if j < len(emotion_labels):
                    all_scores[emotion_labels[j]] = float(score.item())
        
        # scores에서 confidence 추출
        elif hasattr(fer_pred, "scores") and fer_pred.scores:
            # scores가 딕셔너리인 경우
            if isinstance(fer_pred.scores, dict):
                # 가장 높은 점수의 감정 레이블 찾기
                max_score_item = max(fer_pred.scores.items(), key=lambda x: x[1])
                # emotion이 없는 경우 최고 점수의 레이블 사용
                if emotion == "Unknown":
                    emotion = max_score_item[0]
                confidence = float(max_score_item[1])
                all_scores = {k: float(v) for k, v in fer_pred.scores.items()}
    
    return emotion, confidence, all_scores

def save_face_image(face, img, face_id, bbox, emotion, confidence, all_scores, 
                    processed_dir, aligned_dir, filename):
    """
    감지된 얼굴을 이미지로 저장
    
    Parameters:
    - face: 얼굴 객체 (FaceAnalyzer 결과)
    - img: 원본 이미지 (PIL Image)
    - face_id: 얼굴 ID (int)
    - bbox: 얼굴 영역 좌표 [x1, y1, x2, y2]
    - emotion: 감정 레이블 (str)
    - confidence: 감정 신뢰도 (float)
    - all_scores: 모든 감정 점수 (dict)
    - processed_dir: 처리된 이미지 저장 경로
    - aligned_dir: 정렬된 얼굴 저장 경로
    - filename: 파일명 (확장자 제외)
    
    Returns:
    - face_img_path: 저장된 얼굴 이미지 경로
    """
    face_filename = f"{filename}_face{face_id}.jpg"
    face_img_path = os.path.join(processed_dir, face_filename)
    aligned_path = os.path.join(aligned_dir, face_filename)
    
    try:
        # 얼굴 텐서가 유효한지 확인
        if face.tensor is not None and face.tensor.numel() > 0:
            # 1. 정렬된 원본 이미지 저장 (aligned_faces 폴더)
            aligned_pil = transforms.ToPILImage()(face.tensor.squeeze(0))
            aligned_pil.save(aligned_path)
            
            # 2. 감정 정보를 추가한 이미지 저장 (processed_images 폴더)
            face_img = transforms.ToPILImage()(face.tensor.squeeze(0))
            face_img = add_emotion_overlay(face_img, emotion, confidence, all_scores)
            face_img.save(face_img_path)
            
            return face_img_path
        else:
            # 텐서가 유효하지 않은 경우 원본 이미지에서 크롭
            logger.warning(f"얼굴 #{face_id}의 텐서가 유효하지 않습니다. 원본 이미지에서 크롭을 시도합니다.")
            
            # 원본 이미지에서 직접 얼굴 영역 크롭
            img_width, img_height = img.size
            
            # 좌표가 이미지 경계를 넘지 않도록 조정
            x1 = max(0, min(bbox[0], img_width-1))
            y1 = max(0, min(bbox[1], img_height-1))
            x2 = max(0, min(bbox[2], img_width-1))
            y2 = max(0, min(bbox[3], img_height-1))
            
            # 확인: 너무 작은 영역이거나 비정상 좌표인지
            min_size = 20  # 최소 크기
            if x2 > x1 and y2 > y1 and (x2 - x1) > min_size and (y2 - y1) > min_size:
                # 얼굴 영역 크롭
                face_crop = img.crop((x1, y1, x2, y2))
                
                # 정렬된 얼굴 이미지 저장
                face_crop.save(aligned_path)
                
                # 감정 정보가 포함된 이미지 생성 및 저장
                face_img = add_emotion_overlay(face_crop, emotion, confidence, all_scores)
                face_img.save(face_img_path)
                
                return face_img_path
    except Exception as e:
        logger.error(f"얼굴 이미지 처리 오류: {str(e)}")
    
    # 모든 시도 실패 시 None 반환
    return None

def add_emotion_overlay(img, emotion, confidence, all_scores):
    """
    이미지에 감정 정보 오버레이 추가
    """
    # 이미지 복사하여 작업
    face_img = img.copy()
    draw = ImageDraw.Draw(face_img)
    
    # 이미지 크기
    img_width, img_height = face_img.size
    
    # 1. 주요 감정 및 신뢰도 (좌상단)
    main_text = f"{emotion} ({confidence:.2f})"
    
    # 텍스트 위치와 색상 설정
    main_text_position = (10, 10)  # 좌상단에서 약간 띄움
    main_text_color = (255, 0, 0)  # 빨간색
    
    # 폰트 설정 (기본 폰트 사용)
    try:
        # Windows 기본 폰트 시도
        main_font = ImageFont.truetype("arial.ttf", 24)
        detail_font = ImageFont.truetype("arial.ttf", 14)  # 상세 점수용 작은 폰트
    except IOError:
        # 기본 폰트 사용
        main_font = ImageFont.load_default()
        detail_font = ImageFont.load_default()
    
    # 텍스트 크기 측정
    main_text_width, main_text_height = draw.textsize(main_text, font=main_font) if hasattr(draw, 'textsize') else (len(main_text)*14, 24)
    
    # 배경 그리기
    draw.rectangle(
        [main_text_position[0]-5, main_text_position[1]-5, 
         main_text_position[0]+main_text_width+5, main_text_position[1]+main_text_height+5],
        fill=(0, 0, 0, 180)
    )
    
    # 텍스트 그리기
    draw.text(main_text_position, main_text, fill=main_text_color, font=main_font)
    
    # 2. 모든 감정 점수 표시 (우측 하단)
    if all_scores:
        # 모든 감정 점수 텍스트 생성
        score_lines = []
        for emotion_name, score in all_scores.items():
            score_lines.append(f"{emotion_name}: {score:.4f}")
        
        # 텍스트 크기 측정 (대략적 계산)
        line_height = 16
        max_line_width = max([len(line) * 7 for line in score_lines])  # 대략적인 픽셀 계산
        detail_text_height = len(score_lines) * line_height
        
        # 우측 하단 위치 계산
        detail_text_position = (img_width - max_line_width - 10, img_height - detail_text_height - 10)
        
        # 배경 그리기
        draw.rectangle(
            [detail_text_position[0]-5, detail_text_position[1]-5, 
             img_width-5, img_height-5],
            fill=(0, 0, 0, 160)
        )
        
        # 텍스트 그리기 (줄바꿈 처리)
        y_position = detail_text_position[1]
        for line in score_lines:
            draw.text((detail_text_position[0], y_position), line, fill=(255, 255, 255), font=detail_font)
            y_position += line_height
    
    return face_img

@hydra.main(version_base=None, config_path="./facetorch/conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # 사용자 설정 로드
    user_config = load_config()
    
    # 현재 시간을 기준으로 출력 디렉토리 생성
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # 로깅 설정
    logger.info("===== 얼굴 감정 인식 처리 시작 =====")
    logger.info(f"현재 작업 디렉토리: {os.getcwd()}")
    
    # 설정 확인
    if os.path.exists("./facetorch/conf/config.yaml"):
        logger.info("✅ facetorch 설정 파일 존재 확인")
    else:
        logger.warning("❌ facetorch 설정 파일을 찾을 수 없습니다")
    
    # FaceAnalyzer 초기화
    analyzer = FaceAnalyzer(cfg.analyzer)
    logger.info("FaceAnalyzer 초기화 완료")
    
    # 입력 및 출력 디렉토리 설정
    input_dir = user_config["input"]["directory"]
    base_output_dir = user_config["output"]["base_directory"]
    
    # 타임스탬프 폴더 사용 여부
    if user_config["output"]["use_timestamp_folder"]:
        output_dir = os.path.join(base_output_dir, timestamp)
    else:
        output_dir = base_output_dir
    
    processed_images_dir = os.path.join(output_dir, "processed_images")
    aligned_images_dir = os.path.join(output_dir, "aligned_faces")
    
    # 출력 디렉토리 생성
    os.makedirs(base_output_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(processed_images_dir, exist_ok=True)
    os.makedirs(aligned_images_dir, exist_ok=True)
    
    # 감정 분석 결과를 저장할 CSV 파일 및 JSON 파일 경로
    csv_path = os.path.join(output_dir, "emotion_results.csv")
    json_path = os.path.join(output_dir, "emotion_results.json")
    
    # CSV 파일 헤더 작성
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['image_path', 'face_id', 'emotion', 'confidence', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2'])
    
    # JSON 결과 초기화
    json_results = {
        "timestamp": timestamp,
        "total_images": 0,
        "total_faces": 0,
        "results": []
    }
    
    # 이미지 파일 목록 가져오기
    image_files = []
    extensions = user_config["input"]["file_extensions"]
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, f"*.{ext}")))
        image_files.extend(glob.glob(os.path.join(input_dir, f"*.{ext.upper()}")))
    
    logger.info(f"총 {len(image_files)}개의 이미지 파일을 찾았습니다.")
    json_results["total_images"] = len(image_files)
    
    # 처리 설정
    min_face_size = user_config["processing"]["min_face_size"]
    batch_size = user_config["processing"]["batch_size"]
    confidence_threshold = user_config["processing"]["confidence_threshold"]
    emotion_labels = user_config["emotion_labels"]
    
    # 각 이미지 처리
    total_face_count = 0
    filtered_face_count = 0  # 필터링된 얼굴 수 추적
    emotions_summary = {}  # 감정 분포 요약
    
    for idx, img_path in enumerate(image_files):
        try:
            logger.info(f"처리 중 ({idx+1}/{len(image_files)}): {img_path}")
            
            # 전처리: 이미지를 RGB로 변환하여 채널 문제 해결
            img = Image.open(img_path).convert('RGB')
            
            # 원본 파일명 가져오기
            filename = Path(img_path).stem
            
            # 이미지 처리 결과 경로
            image_output_path = os.path.join(processed_images_dir, f"{filename}.jpg")
            
            # 이미지 처리
            result = analyzer.run(
                image_source=img,
                batch_size=batch_size,
                fix_img_size=False,
                return_img_data=True,
                include_tensors=True,
                path_output=None
            )
            
            # 이미지 결과 데이터 초기화
            image_result = {
                "image_name": filename,
                "image_path": None,
                "faces": []
            }
            
            # 결과 출력 및 저장
            if hasattr(result, 'faces') and result.faces:
                face_count = len(result.faces)
                logger.info(f"  감지된 얼굴 수: {face_count}")
                
                # 유효한 얼굴 수 카운트
                valid_face_count = 0
                
                # CSV 파일 열기
                with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
                    csv_writer = csv.writer(csvfile)
                
                    # 각 얼굴에 대한 감정 인식 결과
                    for i, face in enumerate(result.faces):
                        # 얼굴 위치 정보 가져오기
                        if hasattr(face, 'loc'):
                            bbox = [face.loc.x1, face.loc.y1, face.loc.x2, face.loc.y2]
                            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                            
                            # 작은 얼굴 필터링
                            if w < min_face_size or h < min_face_size:
                                filtered_face_count += 1
                                logger.info(f"  얼굴 #{i+1} - 크기가 너무 작아 건너뜀: {w}x{h} < {min_face_size}")
                                continue
                        else:
                            bbox = [0, 0, 0, 0]
                            logger.info(f"  얼굴 #{i+1} - 위치 정보가 없어 건너뜀")
                            continue
                        
                        # 여기서부터는 크기가 충분한 얼굴만 처리됨
                        valid_face_count += 1
                        total_face_count += 1
                        
                        # 감정 분석 결과
                        emotion, confidence, all_scores = extract_emotion(face, emotion_labels)
                        
                        # 얼굴 이미지 저장
                        face_img_path = save_face_image(face, img, i+1, bbox, emotion, confidence, all_scores, 
                                                       processed_images_dir, aligned_images_dir, filename)
                        
                        # CSV에 결과 추가
                        csv_writer.writerow([image_output_path, i+1, emotion, confidence, bbox[0], bbox[1], bbox[2], bbox[3]])
                        
                        # 얼굴 결과 데이터 생성
                        face_data = {
                            "face_id": i+1,
                            "bbox": bbox,
                            "emotion": emotion,
                            "confidence": confidence,
                            "scores": all_scores,
                            "face_image_path": face_img_path
                        }
                        image_result["faces"].append(face_data)
                        
                        # 감정 요약 업데이트
                        if emotion in emotions_summary:
                            emotions_summary[emotion] += 1
                        else:
                            emotions_summary[emotion] = 1
                
                # 유효한 얼굴이 하나 이상 있을 때만 전체 이미지 저장
                if valid_face_count > 0:
                    # 전체 이미지에 얼굴 표시하여 저장
                    result.path_output = image_output_path
                    if "draw_boxes" in analyzer.utilizers:
                        result = analyzer.utilizers["draw_boxes"].run(result)
                    if "draw_landmarks" in analyzer.utilizers:
                        result = analyzer.utilizers["draw_landmarks"].run(result)
                    if "save" in analyzer.utilizers:
                        result = analyzer.utilizers["save"].run(result)
                    logger.info(f"  저장됨: {image_output_path}")
                    # JSON 결과 경로 설정
                    image_result["image_path"] = image_output_path
                else:
                    logger.info(f"  유효한 얼굴이 없어 이미지를 저장하지 않습니다.")
            else:
                logger.info(f"  얼굴이 감지되지 않았습니다.")
            
            # 이미지 결과를 JSON에 추가
            json_results["results"].append(image_result)
            
        except Exception as e:
            logger.error(f"  오류 발생: {img_path} - {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            # 오류 정보를 JSON에 추가
            error_result = {
                "image_name": Path(img_path).stem,
                "error": str(e)
            }
            json_results["results"].append(error_result)
    
    # 총 감지된 얼굴 수 업데이트
    json_results["total_faces"] = total_face_count
    json_results["emotions_summary"] = emotions_summary
    
    # 감정 분석 결과를 JSON 파일로 저장
    with open(json_path, 'w', encoding='utf-8') as jsonfile:
        json.dump(json_results, jsonfile, ensure_ascii=False, indent=2)
    
    # 이미지 목록을 담은 텍스트 파일 생성 (학습용)
    train_list_path = os.path.join(output_dir, "train_list.txt")
    with open(train_list_path, 'w', encoding='utf-8') as f:
        for result in json_results["results"]:
            if "faces" in result:
                for face in result["faces"]:
                    face_path = face.get("face_image_path")
                    emotion = face.get("emotion")
                    confidence = face.get("confidence", 0)
                    if face_path and emotion and emotion != "Unknown" and confidence > confidence_threshold:
                        f.write(f"{face_path} {emotion}\n")
    
    # 결과 요약 출력
    logger.info("\n처리 결과 요약:")
    logger.info(f"총 처리된 이미지: {len(image_files)}개")
    logger.info(f"총 감지된 얼굴: {total_face_count}개")
    logger.info(f"필터링된 작은 얼굴: {filtered_face_count}개")
    logger.info(f"감정별 얼굴 수:")
    for emotion, count in emotions_summary.items():
        logger.info(f"  - {emotion}: {count}개")
    logger.info(f"\n결과가 다음 위치에 저장되었습니다:")
    logger.info(f"  - 처리된 이미지: {processed_images_dir}")
    logger.info(f"  - 정렬된 얼굴 이미지: {aligned_images_dir}")
    logger.info(f"  - CSV 결과: {csv_path}")
    logger.info(f"  - JSON 결과: {json_path}")
    logger.info(f"  - 학습용 목록: {train_list_path}")
    
    logger.info("\n모든 이미지 처리 완료!")

if __name__ == "__main__":
    main()