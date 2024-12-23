from ultralytics import YOLO
import cv2
import os
import matplotlib.pyplot as plt
import math

# ------------------------- IoU 계산 함수 -------------------------
def calculate_iou(box1, box2):
    # 교차 영역 계산
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area_box1 + area_box2 - intersection
    
    # IoU 반환 (0보다 큰 경우만)
    return intersection / union if union > 0 else 0

# ------------------------- 의자 점유 여부 확인 함수 -------------------------
def is_occupied(person, chair):
    """
    사람이 의자에 앉았는지 여부를 판단하는 함수
    :param person: 사람의 좌표 [x1, y1, x2, y2]
    :param chair: 의자의 좌표 [x1, y1, x2, y2]
    :return: 의자 점유 여부 (True/False)
    """
    iou = calculate_iou(person, chair)

    # 사람과 의자 중심 좌표 계산
    person_center_x = (person[0] + person[2]) // 2
    person_center_y = (person[1] + person[3]) // 2
    chair_center_x = (chair[0] + chair[2]) // 2
    chair_center_y = (chair[1] + chair[3]) // 2

    # 유클리드 거리 계산
    distance = math.sqrt((person_center_x - chair_center_x) ** 2 + (person_center_y - chair_center_y) ** 2)
    
    # 세로 겹침 비율 계산
    vertical_overlap = max(0, min(person[3], chair[3]) - max(person[1], chair[1]))
    person_height = person[3] - person[1]
    overlap_ratio = vertical_overlap / person_height if person_height > 0 else 0

    # 점유 조건: IoU, 유클리드 거리, 세로 겹침 비율을 모두 고려
    if (iou > 0.1 or distance < 50) and overlap_ratio > 0.3:
        return True
    return False

# ------------------------- 모델 로드 -------------------------
model = YOLO("yolo11n.pt")  # YOLO 모델 파일 로드

# ------------------------- 이미지 폴더 설정 -------------------------
image_folder = "data"
output_folder = "output"  # 결과 이미지를 저장

# 결과 저장 폴더가 없다면 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# ------------------------- 이미지 파일 목록 -------------------------
# 폴더에 있는 이미지 파일 목록 가져오기
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

if not image_files:
    print("폴더에 이미지 파일이 없습니다.")
else:
    # ------------------------- 이미지 처리 -------------------------
    for image_name in image_files:
        image_path = os.path.join(image_folder, image_name)
        image = cv2.imread(image_path)  # 이미지 읽기
        
        if image is None:
            print(f"{image_name} 이미지를 불러올 수 없습니다.")
            continue

        results = model(image)  # YOLO 모델을 사용해 객체 탐지
        people_boxes, chair_boxes, occupied_chairs = [], [], []

        # ------------------------- 객체 필터링 -------------------------
        for result in results:
            for box in result.boxes:
                xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2] 형태의 좌표
                label = model.names[int(box.cls[0])]  # 객체 라벨
                conf = box.conf[0]  # 신뢰도
                
                if conf > 0.35:  # 신뢰도가 35% 이상인 객체만 처리
                    if label == "person":
                        people_boxes.append(xyxy)  # 사람 박스 추가
                    elif label == "chair":
                        chair_boxes.append(xyxy)  # 의자 박스 추가

        # ------------------------- 의자 점유 여부 확인 -------------------------
        for chair in chair_boxes:
            for person in people_boxes:
                if is_occupied(person, chair):  # 사람이 의자에 앉았으면
                    occupied_chairs.append(chair)
                    break

        # ------------------------- 결과 출력 -------------------------
        total_chairs = len(chair_boxes)
        occupied_seats = len(occupied_chairs)
        available_chairs = total_chairs - occupied_seats

        print(f"\n{image_name} - 좌석 감지 결과:")
        print(f"전체 의자 수: {total_chairs}")
        print(f"현재 차 있는 좌석 수: {occupied_seats}")
        print(f"현재 잔여 좌석 수: {available_chairs}")

        # ------------------------- 결과 이미지 주석 -------------------------
        annotated_image = image.copy()
        for chair in chair_boxes:
            color = (0, 255, 0)  # 의자는 초록색으로 표시
            if chair in occupied_chairs:
                color = (0, 0, 255)  # 점유된 의자는 빨간색으로 표시
            cv2.rectangle(annotated_image, (int(chair[0]), int(chair[1])), (int(chair[2]), int(chair[3])), color, 2)

        # ------------------------- 잔여 좌석 정보 추가 -------------------------
        cv2.putText(annotated_image, f"Available Chairs: {available_chairs}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # ------------------------- 결과 이미지 저장 -------------------------
        annotated_image_path = os.path.join(output_folder, f"annotated_{image_name}")
        cv2.imwrite(annotated_image_path, annotated_image)  # 결과 이미지 저장

        # ------------------------- 결과 이미지 출력 -------------------------
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title(f"Processed Image: {image_name}")
        plt.show()
