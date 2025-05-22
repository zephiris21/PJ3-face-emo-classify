import os
import glob
import csv
import json
from pathlib import Path
import hydra
from omegaconf import DictConfig
from facetorch import FaceAnalyzer
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
import shutil
from torchvision import transforms

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

@hydra.main(version_base=None, config_path="./facetorch/conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # 현재 시간을 기준으로 출력 디렉토리 생성
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # FaceAnalyzer 초기화 (설정 변경은 analyzer.run에서 직접 전달)
    analyzer = FaceAnalyzer(cfg.analyzer)
    print("FaceAnalyzer 초기화 완료")
    
    # 설정 확인 (디버그용)
    has_align_predictor = "align" in cfg.analyzer.predictor
    has_align_utilizer = "align" in cfg.analyzer.utilizer
    has_landmarks_utilizer = "draw_landmarks" in cfg.analyzer.utilizer
    
    print(f"설정 확인: align predictor={has_align_predictor}, align utilizer={has_align_utilizer}, draw_landmarks={has_landmarks_utilizer}")
    
    # 입력 및 출력 디렉토리 설정
    input_dir = "./crawling/파묘"
    base_output_dir = "./outputs"
    output_dir = os.path.join(base_output_dir, timestamp)
    processed_images_dir = os.path.join(output_dir, "processed_images")
    
    # 출력 디렉토리 생성
    os.makedirs(base_output_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(processed_images_dir, exist_ok=True)
    
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
    
    # 이미지 파일 목록 가져오기 (jpg, jpeg, png)
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    
    print(f"총 {len(image_files)}개의 이미지 파일을 찾았습니다.")
    json_results["total_images"] = len(image_files)
    
    # 각 이미지 처리
    total_face_count = 0
    
    # 감정 레이블 매핑 (FER 모델의 인덱스 순서에 따라 조정 필요할 수 있음)
    emotion_labels = [
        "Anger", "Contempt", "Disgust", "Fear", 
        "Happiness", "Neutral", "Sadness", "Surprise"
    ]
    
    for idx, img_path in enumerate(image_files):
        try:
            print(f"처리 중 ({idx+1}/{len(image_files)}): {img_path}")
            
            # 전처리: 이미지를 RGB로 변환하여 채널 문제 해결
            img = Image.open(img_path).convert('RGB')
            
            # 임시 RGB 이미지 저장
            temp_rgb_path = os.path.join(os.path.dirname(img_path), f"temp_rgb_{Path(img_path).name}")
            img.save(temp_rgb_path)
            
            # 원본 파일명 가져오기
            filename = Path(img_path).stem
            
            # 이미지 처리 결과 경로
            image_output_path = os.path.join(processed_images_dir, f"{filename}.jpg")
            
            # 이미지 처리
            result = analyzer.run(
                image_source=temp_rgb_path,  # RGB로 변환된 이미지 사용
                batch_size=1,                # 더 정확한 처리를 위해 배치 크기 축소
                fix_img_size=False,          # 원본 비율 유지
                return_img_data=True,        # 이미지 데이터 반환
                include_tensors=True,        # 텐서 포함 (face.img 및 logits 가져오기 위함)
                path_output=image_output_path
            )
            
            # 이미지 전처리에 대한 정보 출력
            print(f"  이미지 크기: {Image.open(temp_rgb_path).size}, 처리 방식: fix_img_size={False}")
            
            # 임시 파일 삭제
            if os.path.exists(temp_rgb_path):
                os.remove(temp_rgb_path)
            
            # 이미지 결과 데이터 초기화
            image_result = {
                "image_name": filename,
                "image_path": image_output_path,
                "faces": []
            }
            
            # 결과 출력 및 저장
            if hasattr(result, 'faces') and result.faces:
                face_count = len(result.faces)
                total_face_count += face_count
                print(f"  감지된 얼굴 수: {face_count}")
                
                # 얼굴별 결과를 CSV에 추가
                with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    
                    # 각 얼굴에 대한 감정 인식 결과
                    for i, face in enumerate(result.faces):
                        # 얼굴 위치 정보 가져오기 (facetorch는 'loc' 객체에 x1, y1, x2, y2 좌표를 저장)
                        if hasattr(face, 'loc'):
                            bbox = [face.loc.x1, face.loc.y1, face.loc.x2, face.loc.y2]
                        else:
                            bbox = [0, 0, 0, 0]
                        
                        # 감정 분석 결과
                        emotion = "Unknown"
                        confidence = 0.0
                        all_scores = {}
                        face_img_path = None
                        
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
                            
                            # scores에서 감정 확률 추출
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
                        
                        # 얼굴 이미지 저장 (별도 이미지로)
                        face_filename = f"{filename}_face{i+1}.jpg"
                        face_img_path = os.path.join(processed_images_dir, face_filename)
                        
                        # 얼굴 영역 추출 및 저장 방법 변경
                        try:
                            # 1. facetorch 처리 얼굴 이미지 직접 사용 (있을 경우)
                            # Face 객체의 img 필드는 이미 정렬된 얼굴 이미지를 포함
                            if hasattr(face, 'img') and face.img is not None and hasattr(face.img, 'numel') and face.img.numel() > 0:
                                try:
                                    # 텐서를 PIL 이미지로 변환 (채널 첫 번째 차원 제거)
                                    face_img = transforms.ToPILImage()(face.img.squeeze(0))
                                    face_img.save(face_img_path)
                                    print(f"  얼굴 이미지 저장 (facetorch 처리 이미지): {face_img_path}")
                                except Exception as e:
                                    print(f"  facetorch 처리 이미지 변환 오류: {str(e)}")
                                    raise e  # 방법 2로 진행
                            
                            # 2. 원본 이미지에서 직접 얼굴 영역 크롭
                            else:
                                # 원본 이미지에서 얼굴 좌표 기준으로 크롭
                                # 좌표가 원본 이미지 경계 내에 있는지 확인
                                img = Image.open(img_path).convert('RGB')
                                img_width, img_height = img.size
                                
                                # 좌표가 이미지 경계를 넘지 않도록 조정
                                x1 = max(0, min(bbox[0], img_width-1))
                                y1 = max(0, min(bbox[1], img_height-1))
                                x2 = max(0, min(bbox[2], img_width-1))
                                y2 = max(0, min(bbox[3], img_height-1))
                                
                                # 확인: 너무 작은 영역이거나 비정상 좌표인지
                                min_size = 20  # 최소 크기
                                if x2 > x1 and y2 > y1 and (x2 - x1) > min_size and (y2 - y1) > min_size:
                                    face_crop = img.crop((x1, y1, x2, y2))
                                    face_crop.save(face_img_path)
                                    print(f"  얼굴 크롭 저장 (원본 이미지에서): {face_img_path} - 좌표: [{x1}, {y1}, {x2}, {y2}]")
                                else:
                                    print(f"  유효하지 않은 얼굴 좌표: {bbox}, 크기: {x2-x1}x{y2-y1}, 원본 이미지 크기: {img_width}x{img_height}")
                                    # 얼굴 이미지를 저장할 수 없는 경우, facetorch가 생성한 전체 결과 이미지 사용
                                    shutil.copy(image_output_path, face_img_path)
                                    print(f"  대체 이미지 사용: {face_img_path}")
                        except Exception as e:
                            print(f"  얼굴 이미지 저장 오류: {str(e)}")
                            # 실패 시 facetorch가 생성한 전체 결과 이미지 경로 사용
                            shutil.copy(image_output_path, face_img_path)
                            print(f"  오류로 인한 대체 이미지 사용: {face_img_path}")
                        
                        # CSV에 결과 추가
                        csv_writer.writerow([
                            face_img_path if face_img_path else image_output_path,
                            i+1,
                            emotion,
                            f"{confidence:.4f}",
                            bbox[0], bbox[1], bbox[2], bbox[3]
                        ])
                        
                        # 결과 출력
                        print(f"  얼굴 #{i+1} - 감정: {emotion}, 신뢰도: {confidence:.4f}, 위치: {bbox}")
                        
                        # JSON에 얼굴 데이터 추가
                        face_data = {
                            "face_id": i+1,
                            "bbox": bbox,
                            "emotion": emotion,
                            "confidence": float(f"{confidence:.4f}"),
                            "scores": all_scores,
                            "face_image_path": face_img_path
                        }
                        image_result["faces"].append(face_data)
                
                print(f"  저장됨: {image_output_path}")
            else:
                print(f"  얼굴이 감지되지 않았습니다.")
            
            # 이미지 결과를 JSON에 추가
            json_results["results"].append(image_result)
            
        except Exception as e:
            print(f"  오류 발생: {img_path} - {str(e)}")
            import traceback
            print(traceback.format_exc())
            
            # 오류 정보를 JSON에 추가
            error_result = {
                "image_name": Path(img_path).stem,
                "error": str(e)
            }
            json_results["results"].append(error_result)
    
    # 총 감지된 얼굴 수 업데이트
    json_results["total_faces"] = total_face_count
    
    # 감정 분석 결과 요약
    emotions_summary = {}
    for result in json_results["results"]:
        if "faces" in result:
            for face in result["faces"]:
                emotion = face.get("emotion", "Unknown")
                if emotion in emotions_summary:
                    emotions_summary[emotion] += 1
                else:
                    emotions_summary[emotion] = 1
    
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
                    if face_path and emotion and emotion != "Unknown" and confidence > 0.3:  # 신뢰도가 30% 이상인 경우만 포함
                        f.write(f"{face_path} {emotion}\n")
    
    # 결과 요약 출력
    print("\n처리 결과 요약:")
    print(f"총 처리된 이미지: {len(image_files)}개")
    print(f"총 감지된 얼굴: {total_face_count}개")
    print(f"감정별 얼굴 수:")
    for emotion, count in emotions_summary.items():
        print(f"  - {emotion}: {count}개")
    print(f"\n결과가 다음 위치에 저장되었습니다:")
    print(f"  - 처리된 이미지: {processed_images_dir}")
    print(f"  - CSV 결과: {csv_path}")
    print(f"  - JSON 결과: {json_path}")
    print(f"  - 학습용 목록: {train_list_path}")
    
    print("\n모든 이미지 처리 완료!")

if __name__ == "__main__":
    main()