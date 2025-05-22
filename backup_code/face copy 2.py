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
from datetime import datetime

@hydra.main(version_base=None, config_path="./facetorch/conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # 현재 시간을 기준으로 출력 디렉토리 생성
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # FaceAnalyzer 초기화
    analyzer = FaceAnalyzer(cfg.analyzer)
    
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
                batch_size=8,
                fix_img_size=True,
                return_img_data=True,
                include_tensors=False,
                path_output=image_output_path
            )
            
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
                        # 얼굴 영역 좌표
                        bbox = face.bbox if hasattr(face, 'bbox') else [0, 0, 0, 0]
                        
                        # 감정 분석 결과
                        emotion = "Unknown"
                        confidence = 0.0
                        
                        if "fer" in face.preds:
                            emotion = face.preds["fer"].label
                            
                            # confidence 값 확인
                            if hasattr(face.preds["fer"], "confidence"):
                                confidence = face.preds["fer"].confidence
                            # 대안으로 scores 속성 확인
                            elif hasattr(face.preds["fer"], "scores"):
                                scores = face.preds["fer"].scores
                                confidence = max(scores.values())
                                # 만약 label이 없다면 가장 높은 점수의 감정으로 설정
                                if emotion == "Unknown" and scores:
                                    emotion = max(scores.items(), key=lambda x: x[1])[0]
                        
                        # 개별 얼굴 이미지 저장 (가능한 경우)
                        face_img_path = None
                        if hasattr(face, 'img_cropped') and face.img_cropped is not None:
                            face_filename = f"{filename}_face{i+1}.jpg"
                            face_img_path = os.path.join(processed_images_dir, face_filename)
                            
                            # PIL Image 형식으로 변환 및 저장
                            if isinstance(face.img_cropped, np.ndarray):
                                face_img = Image.fromarray(face.img_cropped)
                                face_img.save(face_img_path)
                        
                        # CSV에 결과 추가
                        csv_writer.writerow([
                            image_output_path if face_img_path is None else face_img_path,
                            i+1,
                            emotion,
                            f"{confidence:.4f}",
                            bbox[0], bbox[1], bbox[2], bbox[3]
                        ])
                        
                        # 결과 출력
                        print(f"  얼굴 #{i+1} - 감정: {emotion}, 신뢰도: {confidence:.4f}")
                        
                        # JSON에 얼굴 데이터 추가
                        face_data = {
                            "face_id": i+1,
                            "bbox": bbox,
                            "emotion": emotion,
                            "confidence": float(f"{confidence:.4f}"),
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
                    if face_path and emotion and emotion != "Unknown":
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