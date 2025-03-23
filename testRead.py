# import cv2
# import torch
# import os
# from ultralytics import YOLO

# # -------------------- CONFIGURATION --------------------
# ocr_model_path = r"bestRead.pt"
# ocr_model = YOLO(ocr_model_path)

# device = "cuda" if torch.cuda.is_available() else "cpu"
# ocr_model.to(device)

# OCR_CONF_THRESHOLD = 0.4

# # -------------------- PHÂN LOẠI BIỂN SỐ --------------------
# def classify_plate_shape(plate_img):
#     h, w = plate_img.shape[:2]
#     aspect_ratio = w / h
#     return "horizontal" if aspect_ratio > 2 else "square"

# # -------------------- SẮP XẾP KÝ TỰ OCR --------------------
# def sort_ocr_results(detected_chars, plate_shape):
#     if plate_shape == "horizontal":
#         return "".join([char for _, _, char in sorted(detected_chars, key=lambda x: x[0])])
    
#     y_values = [y for _, y, _ in detected_chars]
#     y_threshold = (max(y_values) + min(y_values)) / 2
    
#     top_line = sorted([(x, char) for x, y, char in detected_chars if y < y_threshold], key=lambda x: x[0])
#     bottom_line = sorted([(x, char) for x, y, char in detected_chars if y >= y_threshold], key=lambda x: x[0])
    
#     return "".join([char for _, char in top_line]) + " " + "".join([char for _, char in bottom_line])

# # -------------------- NHẬN DIỆN KÝ TỰ --------------------
# def recognize_text(image):
#     plate_shape = classify_plate_shape(image)
#     ocr_results = ocr_model(image)
#     detected_chars = [(int(box.xyxy[0][0]), int(box.xyxy[0][1]), ocr_model.names[int(box.cls[0])])
#                       for result in ocr_results for box in result.boxes if float(box.conf[0]) > OCR_CONF_THRESHOLD]
#     return sort_ocr_results(detected_chars, plate_shape) if detected_chars else "N/A"

# # -------------------- Xử LÝ  ẢNH --------------------
# def process_images_in_folder(image_folder, output_file):
#     if not os.path.exists(image_folder):
#         print(f"❌ Thư mục {image_folder} không tồn tại!")
#         return

#     with open(output_file, "w") as f:
#         for image_name in os.listdir(image_folder):
#             image_path = os.path.join(image_folder, image_name)

#             if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
#                 continue

#             print(f"🔍 Đang xử lý: {image_name}")
#             image = cv2.imread(image_path)

#             if image is None:
#                 print(f"⚠ Không thể đọc ảnh: {image_name}")
#                 continue

#             text = recognize_text(image)
#             f.write(f"{image_name}: {text}\n")

#     print(f"✅ Nhận diện hoàn tất! Kết quả lưu tại: {output_file}")

# # -------------------- CHẠY CHƯƠNG TRÌNH --------------------
# if __name__ == "__main__":
#     image_folder = "detected_plates/"
#     output_file = "output.txt"
#     process_images_in_folder(image_folder, output_file)

import cv2
import numpy as np
import os
from ultralytics import YOLO

# Load mô hình YOLO đã huấn luyện để đọc ký tự
model = YOLO("bestRead.pt")

# Danh sách nhãn ký tự
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
               'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 
               'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 
               'W', 'X', 'Y', 'Z']

def process_plate(image_name, input_folder):
    image_path = os.path.join(input_folder, image_name)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Lỗi: Không thể đọc ảnh từ {image_path}. Bỏ qua...")
        return f"{image_name}: Không nhận diện được"

    # Dự đoán ký tự trên biển số
    results = model(image)
    detected_chars = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            label = class_names[class_id]
            detected_chars.append({"char": label, "x": x1, "y": y1})

    if not detected_chars:
        return f"{image_name}: Không nhận diện được"

    # Xác định số hàng của biển số
    y_coords = [char["y"] for char in detected_chars]
    y_mean = np.mean(y_coords)

    upper_row = [char for char in detected_chars if char["y"] < y_mean]
    lower_row = [char for char in detected_chars if char["y"] >= y_mean]

    upper_row.sort(key=lambda char: char["x"])
    lower_row.sort(key=lambda char: char["x"])

    # Thêm khoảng cách nếu chênh lệch x lớn giữa các ký tự
    def format_plate(chars):
        chars.sort(key=lambda char: char["x"])
        formatted_text = chars[0]["char"]
        for i in range(1, len(chars)):
            if chars[i]["x"] - chars[i - 1]["x"] > 20:  # Ngưỡng để thêm khoảng trắng
                formatted_text += " "
            formatted_text += chars[i]["char"]
        return formatted_text

    # Ghép thành chuỗi biển số có khoảng cách hợp lý
    if len(lower_row) > 0:
        plate_text = format_plate(upper_row) + " " + format_plate(lower_row)
    else:
        plate_text = format_plate(upper_row)

    return plate_text

