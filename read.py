# import cv2
# import os
# from ultralytics import YOLO
# import concurrent.futures

# # Global: load model bestRead (hãy chắc chắn file bestRead.pt nằm cùng thư mục hoặc chỉ đường dẫn đúng)
# device = 'cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'
# best_read_model = YOLO("bestRead.pt").to(device)

# def preprocess_plate_for_read(plate_img):
#     """
#     Thực hiện các bước tiền xử lý nếu cần, ví dụ chuyển sang grayscale, tăng độ tương phản,...
#     Bạn có thể tùy chỉnh thêm cho phù hợp với model bestRead.
#     """
#     # Ví dụ chuyển sang grayscale rồi chuyển về 3 kênh để tương thích với model YOLO
#     gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
#     processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
#     return processed

# def read_plate_text(plate_img):
#     """
#     Dùng model bestRead để dự đoán ký tự trên biển số.
#     Giả sử model trả về kết quả với cấu trúc: result.boxes, result.names,...
#     Bạn có thể cần điều chỉnh tùy vào cách model huấn luyện và trả về kết quả.
#     """
#     processed_plate = preprocess_plate_for_read(plate_img)
#     results = best_read_model(processed_plate, conf=0.5)
    
#     detected_text = ""
#     # Giả sử kết quả chứa 1 box cho toàn bộ biển số, hoặc bạn gộp các box lại
#     for result in results:
#         for box in result.boxes:
#             cls_id = int(box.cls[0])
#             # Dùng tên lớp để làm ký tự, giả sử model huấn luyện để trả về ký tự từng box
#             detected_text += result.names[cls_id]
#     return detected_text.strip()

# def process_plate(plate_file, input_folder):
#     plate_path = os.path.join(input_folder, plate_file)
#     img = cv2.imread(plate_path)
#     if img is None:
#         return plate_file, ""
#     text = read_plate_text(img)
#     print(f"Đã đọc {plate_file}: {text}")
#     return plate_file, text

# def process_detected_plates():
#     input_folder = "detected_plates"
#     result_file = "result.txt"
#     if not os.path.exists(input_folder):
#         print(f"Folder '{input_folder}' không tồn tại!")
#         return

#     plate_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
#     if not plate_files:
#         print("Không tìm thấy ảnh biển số nào!")
#         return

#     results = []
#     # Sử dụng ThreadPoolExecutor để chạy song song các task OCR
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         futures = [executor.submit(process_plate, plate, input_folder) for plate in plate_files]
#         for future in concurrent.futures.as_completed(futures):
#             plate_file, text = future.result()
#             results.append((plate_file, text))

#     with open(result_file, "w", encoding="utf-8") as f:
#         for plate_file, text in results:
#             f.write(f"{plate_file}: {text}\n")
#     print("Kết quả đã được ghi vào result.txt")

# if __name__ == "__main__":
#     process_detected_plates()

# import cv2
# import os
# from ultralytics import YOLO
# import concurrent.futures

# # Load model
# device = 'cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'
# best_read_model = YOLO("bestRead.pt").to(device)

# def preprocess_plate_for_read(plate_img):
#     gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
#     gray = cv2.equalizeHist(gray)
#     gray = cv2.GaussianBlur(gray, (3, 3), 0)
#     _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     processed = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
#     return processed

# def read_plate_text(plate_img):
#     processed_plate = preprocess_plate_for_read(plate_img)
#     results = best_read_model(processed_plate, conf=0.5)
    
#     detected_chars = []
#     for result in results:
#         for box in result.boxes:
#             cls_id = int(box.cls[0])
#             char = result.names[cls_id]
#             x_min = box.xyxy[0][0].item()
#             detected_chars.append((x_min, char))
    
#     detected_chars.sort(key=lambda x: x[0])
#     detected_text = "".join([char for _, char in detected_chars])
#     return detected_text.strip()

# def process_plate(plate_file, input_folder):
#     plate_path = os.path.join(input_folder, plate_file)
#     img = cv2.imread(plate_path)
#     if img is None:
#         return plate_file, ""
#     text = read_plate_text(img)
#     print(f"Đã đọc {plate_file}: {text}")
#     return plate_file, text

# def process_detected_plates():
#     input_folder = "detected_plates"
#     result_file = "result.txt"
#     if not os.path.exists(input_folder):
#         print(f"Folder '{input_folder}' không tồn tại!")
#         return

#     plate_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
#     if not plate_files:
#         print("Không tìm thấy ảnh biển số nào!")
#         return

#     results = []
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         futures = [executor.submit(process_plate, plate, input_folder) for plate in plate_files]
#         for future in concurrent.futures.as_completed(futures):
#             plate_file, text = future.result()
#             results.append((plate_file, text))

#     with open(result_file, "w", encoding="utf-8") as f:
#         for plate_file, text in results:
#             f.write(f"{plate_file}: {text}\n")
#     print("Kết quả đã được ghi vào result.txt")

# if __name__ == "__main__":
#     process_detected_plates()

