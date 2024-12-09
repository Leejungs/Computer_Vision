from ultralytics import YOLO
import cv2

# ------------------------- YOLO 모델 설정 -------------------------
model_path = "models/yolo11n.pt"
image_path = "data/cafe.jpg"

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
        if box.conf[0] > 0.3:  # 신뢰도 기준 적용
            if label == "person":
                people_boxes.append(box.xyxy[0].tolist())
            elif label == "chair":
                chair_boxes.append(box.xyxy[0].tolist())

# ------------------------- 점유된 의자 수 계산 -------------------------
occupied_seats = 0
for chair in chair_boxes:
    for person in people_boxes:
        # 사람과 의자 간 관계 판단: 바운딩 박스 전체 비교
        if (person[0] < chair[2] and person[2] > chair[0] and  # 좌우 겹침
                person[1] < chair[3] and person[3] > chair[1]):  # 상하 겹침
            occupied_seats += 1
            break  # 이미 점유된 의자는 중복 계산 방지

# ------------------------- 결과 계산 -------------------------
total_chairs = len(chair_boxes)
available_chairs = total_chairs - occupied_seats

# ------------------------- 결과 출력 -------------------------
print("\n좌석 감지 결과:")
print(f"전체 의자 수: {total_chairs}")
print(f"현재 차 있는 좌석 수: {occupied_seats}")
print(f"현재 잔여 좌석 수: {available_chairs}")

# ------------------------- 시각화 -------------------------
annotated_image = image.copy()
for chair in chair_boxes:
    color = (0, 255, 0)  # 초록색
    cv2.rectangle(annotated_image, (int(chair[0]), int(chair[1])), (int(chair[2]), int(chair[3])), color, 2)  # 의자
for person in people_boxes:
    color = (255, 0, 0)  # 파란색
    cv2.rectangle(annotated_image, (int(person[0]), int(person[1])), (int(person[2]), int(person[3])), color, 2)  # 사람

cv2.putText(annotated_image, f"Available Chairs: {available_chairs}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

cv2.imshow("Annotated Image", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
