# import cv2
# import torch
# import os
# import numpy as np
# from ultralytics import YOLO
# from sort.sort import Sort  # Đảm bảo bạn đã cài đặt hoặc copy module SORT

# # Cấu hình đường dẫn
# detect_model_path = "bestDetect.pt"  # Model detect biển số
# video_path = "okok1080.mp4"          # Video nguồn
# output_folder = "detected_plates"    # Folder lưu ảnh biển số

# # Sử dụng GPU nếu có
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(f"Using device: {device}")

# # Load model YOLO
# detect_model = YOLO(detect_model_path).to(device)

# # Tạo folder lưu ảnh nếu chưa có
# os.makedirs(output_folder, exist_ok=True)

# # Khởi tạo tracker SORT
# tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

# # Mở video
# cap = cv2.VideoCapture(video_path)
# if not cap.isOpened():
#     print("Không mở được video!")
#     exit()

# frame_count = 0
# plate_index = 0
# tracked_plate_ids = set()  # Set lưu lại các track_id đã lưu

# print("Bắt đầu xử lý video với tracking biển số...")

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#     frame_count += 1

#     # Detect biển số qua YOLO
#     results = detect_model(frame, conf=0.5)
#     detections = []
#     for result in results:
#         for box in result.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             conf = float(box.conf[0])
#             # Chỉ thêm các box có kích thước hợp lý (nếu cần lọc thêm)
#             detections.append([x1, y1, x2, y2, conf])
#     detections = np.array(detections)

#     # Cập nhật tracker SORT với các detection hiện tại
#     if detections.shape[0] > 0:
#         tracked_objects = tracker.update(detections)
#     else:
#         tracked_objects = np.empty((0, 5))

#     # Duyệt qua các đối tượng đã được theo dõi
#     for obj in tracked_objects:
#         x1, y1, x2, y2, track_id = obj
#         track_id = int(track_id)
#         # Nếu track_id này đã lưu rồi thì bỏ qua
#         if track_id in tracked_plate_ids:
#             continue
#         # Đánh dấu track_id mới
#         tracked_plate_ids.add(track_id)

#         # Ép lại về số nguyên cho bounding box
#         x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
#         # Cắt ảnh biển số
#         plate_img = frame[y1:y2, x1:x2].copy()
#         if plate_img.size == 0:
#             continue

#         # Lưu ảnh biển số
#         save_path = os.path.join(output_folder, f"plate_{plate_index}.jpg")
#         cv2.imwrite(save_path, plate_img)
#         print(f"Frame {frame_count}: Lưu biển số {plate_index} với track ID {track_id}")
#         plate_index += 1

#     # Hiển thị tạm (có thể comment nếu không cần)
#     cv2.imshow("Video", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
# print("Xong! Check folder 'detected_plates' để xem kết quả nhé!")

# import cv2
# import torch
# import os
# import numpy as np
# from ultralytics import YOLO
# from sort.sort import Sort  # Đảm bảo rằng thư mục sort có __init__.py

# # Cấu hình đường dẫn
# detect_model_path = "bestDetect.pt"   # Model detect biển số
# video_path = "okok1080.mp4"           # Video nguồn
# output_folder = "detected_plates"     # Folder lưu ảnh biển số

# # Sử dụng GPU nếu có
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(f"Using device: {device}")

# # Load model YOLO
# detect_model = YOLO(detect_model_path).to(device)

# # Tạo folder lưu ảnh nếu chưa có
# os.makedirs(output_folder, exist_ok=True)

# # Khởi tạo tracker SORT
# tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

# # Mở video
# cap = cv2.VideoCapture(video_path)
# if not cap.isOpened():
#     print("Không mở được video!")
#     exit()

# frame_count = 0
# plate_index = 0
# tracked_plate_ids = set()  # Set lưu lại các track_id đã lưu

# print("Bắt đầu xử lý video với tracking biển số...")

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#     frame_count += 1

#     # Detect biển số qua YOLO
#     results = detect_model(frame, conf=0.5)
#     detections = []
#     for result in results:
#         for box in result.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             conf = float(box.conf[0])
#             detections.append([x1, y1, x2, y2, conf])
#     detections = np.array(detections)

#     # Cập nhật tracker SORT với các detection hiện tại
#     if detections.shape[0] > 0:
#         tracked_objects = tracker.update(detections)
#     else:
#         tracked_objects = np.empty((0, 5))

#     # Duyệt qua các đối tượng đã được theo dõi
#     for obj in tracked_objects:
#         x1, y1, x2, y2, track_id = obj
#         track_id = int(track_id)
#         # Nếu track_id này đã lưu rồi thì bỏ qua
#         if track_id in tracked_plate_ids:
#             continue
#         # Đánh dấu track_id mới
#         tracked_plate_ids.add(track_id)

#         # Ép lại về số nguyên cho bounding box
#         x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
#         # Cắt ảnh biển số
#         plate_img = frame[y1:y2, x1:x2].copy()
#         if plate_img.size == 0:
#             continue

#         # Lưu ảnh biển số
#         save_path = os.path.join(output_folder, f"plate_{plate_index}.jpg")
#         cv2.imwrite(save_path, plate_img)
#         print(f"Frame {frame_count}: Lưu biển số {plate_index} với track ID {track_id}")
#         plate_index += 1

#     # Hiển thị tạm (có thể comment nếu không cần)
#     cv2.imshow("Video", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
# print("Xong! Check folder 'detected_plates' để xem kết quả nhé!")

# # Sau khi xử lý xong video, gọi module read để đọc các ảnh và ghi kết quả vào result.txt
# import read
# read.process_detected_plates()

# print("Chương trình hoàn thành, tạm biệt!")

import cv2
import torch
import os
import numpy as np
from ultralytics import YOLO
from sort.sort import Sort  # Đảm bảo rằng thư mục sort có __init__.py
import testRead  # Import module read.py

# Cấu hình đường dẫn
detect_model_path = "bestDetect.pt"   # Model detect biển số
video_path = "test.mp4"               # Video nguồn
output_folder = "detected_plates"     # Folder lưu ảnh biển số

# Sử dụng GPU nếu có
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load model YOLO
detect_model = YOLO(detect_model_path).to(device)

# Tạo folder lưu ảnh nếu chưa có
os.makedirs(output_folder, exist_ok=True)

# Mở file kết quả để ghi
result_file = "result.txt"
with open(result_file, "w", encoding="utf-8") as f:
    pass  # Tạo file trống

# Khởi tạo tracker SORT
tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

# Mở video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Không mở được video!")
    exit()

frame_count = 0
plate_index = 0
tracked_plate_ids = set()  # Set lưu lại các track_id đã lưu

print("Bắt đầu xử lý video với tracking biển số...")

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

        # Lưu ảnh biển số
        plate_filename = f"plate_{plate_index}.jpg"
        save_path = os.path.join(output_folder, plate_filename)
        cv2.imwrite(save_path, plate_img)
        
        # Đọc biển số vừa lưu bằng module read.py
        plate_text = testRead.process_plate(plate_filename, output_folder)
        
        # Lưu kết quả vào file
        with open(result_file, "a", encoding="utf-8") as f:
            f.write(f"{plate_filename}: {plate_text}\n")
        
        print(f"Frame {frame_count}: Lưu biển số {plate_index} với track ID {track_id}, kết quả: {plate_text}")
        plate_index += 1

    # Hiển thị tạm (có thể comment nếu không cần)
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Xong! Check folder 'detected_plates' và file 'result.txt' để xem kết quả nhé!")


