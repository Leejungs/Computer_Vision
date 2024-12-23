### 김다현, 이정수, 최소원, 윤채영
# 실시간으로 공공장소 빈 좌석 확인하기
<img src="Image_detection_people.png" width="600" height="400"/>

## 프로젝트 개요
### 개발 목적
공공장소(예: 카페, 도서관)의 좌석 여부를 감지하여 사용자에게 빈 좌석 정보를 제공합니다. <br/>
경량화된 YOLO11n 딥러닝 모델을 활용해 좌석 점유 여부를 효율적으로 분석하며, 저비용 장비로 경제적이고 간편하게 구현할 수 있습니다.

### 기대 효과
- **사용자 편의성 향상**: 빈 좌석을 빠르게 찾아 시간을 절약.
- **운영 효율성 개선**: 관리자에게 좌석 사용 데이터를 제공하여 공간 활용도 최적화.
- **경제적 구현 가능성**: 기존 카메라 장비를 활용해 추가 비용 절감.

---

## 주요 기능
### 1. 좌석 점유 상태 감지
- 데스크탑, 노트북 등에 탑재되어 있는 웹캠(카메라)를 활용해 좌석에 사람이 앉아있는지를 인식.
- YOLO11n 모델로 사람만을 탐지하여 좌석 상태를 "사용 중"또는 "비어 있음"으로 구분.

### 2. 정보 제공
- 빈 좌석 데이터를 사용자에게 API 또는 로컬 기록 형태로 제공합니다.
- 관리자에게는 시간별로 좌석을 사용한 기록들이 제공될 수 있도록 하여 또 다른 활용 데이터로도 활용 가능합니다. 

### 3. 경량화 모델
- YOLO11n 모델을 최적화하여 실시간으로 분석이 가능.
- 저사양 장비에서도 높은 성능을 유지하며 원할이 작동이 가능.

---

## 프로젝트 실행 방법
### 1. 환경 설정
- Python 버전: 3.11.3
- 필요 라이브러리 설치:
  ```bash
  pip install -r requirements.txt
  ```

### 2. 실행 명령어
- 사진으로 테스트:
  ```bash
  python seat_detector.py --image test_image.jpg
  ```
- 웹캠으로 테스트:
  ```bash
  python seat_detector.py --webcam
  ```
---

## 프로젝트 구조
```plaintext
project/
├── models/                # YOLO11n 모델 파일
├── utils/                 # 보조 스크립트 및 설정 파일
├── data/                  # 테스트용 동영상 및 이미지
├── output_folder/   #테스트 결과 동영상 및 이미지
├── seat_detector.py       # 메인 실행 스크립트
├── requirements.txt       # 필요한 라이브러리 목록
└── README.md              # 프로젝트 설명 파일
```

---

## 시뮬레이션 결과
### 시연 이미지
![annotated_test8](https://github.com/user-attachments/assets/0f98df34-098e-4075-978d-37d68507cc67)
![annotated_cafe](https://github.com/user-attachments/assets/e981e858-5e28-4ed9-b488-8f91bfc694c1)



---

## 기술 상세
### YOLO11n 경량화 모델
- YOLO11n은 딥러닝 모델 YOLO(You Only Look Once)를 경량화하여 실시간 객체 탐지 성능을 최적화한 버전입니다.
- 빠른 탐지 속도와 낮은 메모리 사용량을 통해 기존 하드웨어에서도 효율적으로 작동합니다.

### 주요 구현 기술
1. **객체 감지 및 추적**:
   - YOLO11n을 이용해 객체(사람)를 감지하고, 좌석의 점유 상태를 확인합니다.
2. **데이터 로깅**:
   - 좌석 점유 데이터를 시간별로 기록하여 관리자용 분석 자료를 제공합니다.
3. **사용자 인터페이스**:
   - API 또는 로그 파일 형태로 결과를 출력하여 활용도를 높입니다.

---

## 개발 일정 및 역할 분담
### 역할 분담
- **개발 총괄**: YOLO11n 모델 튜닝 및 좌석 점유 여부 분석 구현.
- **테스트**: 다양한 환경(사진, 실시간 웹캠)에서 결과 검증.
- **문서화**: 프로젝트 문서 작성 및 결과 시각화.

---

## 참고 자료
1. [YOLO 논문](https://arxiv.org/abs/1506.02640)
2. [YOLOv4 및 YOLOv5 모델 개요](https://github.com/AlexeyAB/darknet)
3. [PyTorch 공식 문서](https://pytorch.org/)

---

본 프로젝트는 공공장소 좌석 점유 여부 감지를 위한 경량화된 AI 솔루션을 제공합니다. 향후 API 개발 및 클라이언트 애플리케이션과의 통합을 통해 확장 가능합니다.
