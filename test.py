import cv2
import torch
import os
import numpy as np
from ultralytics import YOLO
from sort.sort import Sort  # Đảm bảo rằng thư mục sort có __init__.py

# Cấu hình đường dẫn
detect_model_path = "bestDetect.pt"   # Model detect biển số
read_model_path = "bestRead.pt"       # Model đọc biển số
video_path = "okok1080.mp4"           # Video nguồn
output_folder = "detected_plates"     # Folder lưu ảnh biển số
result_file = "result.txt"            # File kết quả

# Sử dụng GPU nếu có
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load model YOLO cho cả phát hiện và đọc biển số
detect_model = YOLO(detect_model_path).to(device)
read_model = YOLO(read_model_path).to(device)

# Tạo folder lưu ảnh nếu chưa có
os.makedirs(output_folder, exist_ok=True)

# Khởi tạo tracker SORT
tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

# Hàm tiền xử lý ảnh biển số trước khi đọc
def preprocess_plate_for_read(plate_img):
    """
    Thực hiện các bước tiền xử lý nếu cần, ví dụ chuyển sang grayscale, tăng độ tương phản,...
    """
    # Ví dụ chuyển sang grayscale rồi chuyển về 3 kênh để tương thích với model YOLO
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return processed

# Hàm đọc biển số
def read_plate_text(plate_img):
    """
    Dùng model bestRead để dự đoán ký tự trên biển số.
    """
    processed_plate = preprocess_plate_for_read(plate_img)
    results = read_model(processed_plate, conf=0.5)
    
    detected_text = ""
    # Giả sử kết quả chứa 1 box cho toàn bộ biển số, hoặc bạn gộp các box lại
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            # Dùng tên lớp để làm ký tự, giả sử model huấn luyện để trả về ký tự từng box
            detected_text += result.names[cls_id]
    return detected_text.strip()

# Mở file kết quả để ghi
result_log = open(result_file, "w", encoding="utf-8")

# Mở video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Không mở được video!")
    result_log.close()
    exit()

frame_count = 0
plate_index = 0
tracked_plate_ids = set()  # Set lưu lại các track_id đã lưu

print("Bắt đầu xử lý video với tracking và đọc biển số theo thời gian thực...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # Detect biển số qua YOLO
    results = detect_model(frame, conf=0.5)
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            detections.append([x1, y1, x2, y2, conf])
    detections = np.array(detections)

    # Cập nhật tracker SORT với các detection hiện tại
    if detections.shape[0] > 0:
        tracked_objects = tracker.update(detections)
    else:
        tracked_objects = np.empty((0, 5))

    # Duyệt qua các đối tượng đã được theo dõi
    for obj in tracked_objects:
        x1, y1, x2, y2, track_id = obj
        track_id = int(track_id)
        # Nếu track_id này đã lưu rồi thì bỏ qua
        if track_id in tracked_plate_ids:
            continue
        # Đánh dấu track_id mới
        tracked_plate_ids.add(track_id)

        # Ép lại về số nguyên cho bounding box
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        # Cắt ảnh biển số
        plate_img = frame[y1:y2, x1:x2].copy()
        if plate_img.size == 0:
            continue

        # Đặt tên cho file ảnh biển số
        plate_filename = f"plate_{plate_index}.jpg"
        save_path = os.path.join(output_folder, plate_filename)
        
        # Lưu ảnh biển số
        cv2.imwrite(save_path, plate_img)
        
        # Đọc biển số ngay lập tức
        plate_text = read_plate_text(plate_img)
        
        # Ghi kết quả vào file
        result_log.write(f"{plate_filename}: {plate_text}\n")
        result_log.flush()  # Đảm bảo dữ liệu được ghi ngay lập tức
        
        print(f"Frame {frame_count}: Biển số {plate_index} (ID {track_id}): {plate_text}")
        plate_index += 1

    # Hiển thị tạm (có thể comment nếu không cần)
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
result_log.close()
print(f"Xong! Đã xử lý {plate_index} biển số. Kết quả được lưu vào '{result_file}'")