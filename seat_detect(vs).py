from ultralytics import YOLO
import cv2
import math

# ------------------------- YOLO 모델 설정 -------------------------
model_path = "models/yolo11n.pt"
image_path = "C:/Users/choil/OneDrive/Desktop/cv_project/data/cafe.jpg"

model = YOLO(model_path)

# ------------------------- 이미지 읽기 -------------------------
image = cv2.imread(image_path)
if image is None:
    raise ValueError("이미지를 불러올 수 없습니다.")

# ------------------------- 객체 감지 수행 -------------------------
results = model(image)

# ------------------------- 객체 필터링 -------------------------
people_boxes = []
chair_boxes = []

for result in results:
    for box in result.boxes:
        label = model.names[int(box.cls[0])]
        if box.conf[0] > 0.35:  # 신뢰도 기준 적용
            if label == "person":
                people_boxes.append(box.xyxy[0].tolist())
            elif label == "chair":
                chair_boxes.append(box.xyxy[0].tolist())

# ------------------------- 점유된 의자 수 계산 -------------------------
occupied_chairs = []
for chair in chair_boxes:
    for person in people_boxes:
        # 사람 중심 좌표
        person_center_x = (person[0] + person[2]) // 2
        person_center_y = (person[1] + person[3]) // 2

        # 의자 중심 좌표
        chair_center_x = (chair[0] + chair[2]) // 2
        chair_center_y = (chair[1] + chair[3]) // 2

        # 유클리드 거리 계산
        distance = math.sqrt((person_center_x - chair_center_x) ** 2 + (person_center_y - chair_center_y) ** 2)

        # 세로 겹침 비율 계산
        vertical_overlap = max(0, min(person[3], chair[3]) - max(person[1], chair[1]))
        person_height = person[3] - person[1]
        overlap_ratio = vertical_overlap / person_height if person_height > 0 else 0

        # 가로와 세로 겹침 확인
        # 가로는 비율까지 계산해 비교할 이유 없음
        if (chair[0] < person[2] and chair[2] > person[0]) and (chair[1] < person[3] and chair[3] > person[1]):
            # 점유 조건: 거리 또는 세로 겹침 비율
            if distance < 30 or overlap_ratio > 0.3:
                occupied_chairs.append(chair)
                break

# ------------------------- 결과 계산 -------------------------
total_chairs = len(chair_boxes)
occupied_seats = len(occupied_chairs)
available_chairs = total_chairs - occupied_seats

# ------------------------- 결과 출력 -------------------------
print("\n좌석 감지 결과:")
print(f"전체 의자 수: {total_chairs}")
print(f"현재 차 있는 좌석 수: {occupied_seats}")
print(f"현재 잔여 좌석 수: {available_chairs}")

# ------------------------- 시각화 -------------------------
annotated_image = image.copy()
for chair in chair_boxes:
    color = (0, 255, 0)  # 기본 색상: 초록 (빈 의자)
    if chair in occupied_chairs:
        color = (255, 0, 0)  # 점유된 의자: 파랑
    cv2.rectangle(annotated_image, (int(chair[0]), int(chair[1])), (int(chair[2]), int(chair[3])), color, 2)  # 의자

# for person in people_boxes:
    # cv2.rectangle(annotated_image, (int(person[0]), int(person[1])), (int(person[2]), int(person[3])), (0, 255, 255), 2)  # 노란색

cv2.putText(annotated_image, f"available_chairs: {available_chairs}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

cv2.imshow("Annotated Image", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
