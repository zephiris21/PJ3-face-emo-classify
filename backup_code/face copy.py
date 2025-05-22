import os
import glob
from pathlib import Path
import hydra
from omegaconf import DictConfig
from facetorch import FaceAnalyzer
from PIL import Image
import numpy as np

@hydra.main(version_base=None, config_path="./facetorch/conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # FaceAnalyzer 초기화
    analyzer = FaceAnalyzer(cfg.analyzer)
    
    # 입력 및 출력 디렉토리 설정
    input_dir = "./inputs"
    output_dir = "./processed"
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 이미지 파일 목록 가져오기 (jpg, jpeg, png)
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    
    print(f"총 {len(image_files)}개의 이미지 파일을 찾았습니다.")
    
    # 각 이미지 처리
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
            output_path = os.path.join(output_dir, f"{filename}_processed.jpg")
            
            # 이미지 처리
            result = analyzer.run(
                image_source=temp_rgb_path,  # RGB로 변환된 이미지 사용
                batch_size=8,
                fix_img_size=True,
                return_img_data=True,
                include_tensors=False,
                path_output=output_path
            )
            
            # 임시 파일 삭제
            if os.path.exists(temp_rgb_path):
                os.remove(temp_rgb_path)
            
            # 결과 출력
            if hasattr(result, 'faces') and result.faces:
                face_count = len(result.faces)
                print(f"  감지된 얼굴 수: {face_count}")
                
                # 각 얼굴에 대한 감정 인식 결과
                for i, face in enumerate(result.faces):
                    if "fer" in face.preds:
                        emotion = face.preds["fer"].label
                        
                        # confidence 속성이 있는지 확인
                        confidence = "Unknown"
                        if hasattr(face.preds["fer"], "confidence"):
                            confidence = f"{face.preds['fer'].confidence:.4f}"
                        # 대안으로 scores 속성 확인
                        elif hasattr(face.preds["fer"], "scores"):
                            highest_score = max(face.preds["fer"].scores.values())
                            confidence = f"{highest_score:.4f}"
                        
                        print(f"  얼굴 #{i+1} - 감정: {emotion}, 신뢰도: {confidence}")
                
                print(f"  저장됨: {output_path}")
            else:
                print(f"  얼굴이 감지되지 않았습니다.")
            
        except Exception as e:
            print(f"  오류 발생: {img_path} - {str(e)}")
    
    print("모든 이미지 처리 완료!")

if __name__ == "__main__":
    main()