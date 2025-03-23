import cv2
import torch
import numpy as np
import os
import threading
from ultralytics import YOLO

# -------------------- CONFIGURATION --------------------
plate_model_path = r"D:/KNN/Cắt ảnh/Cắt ảnh/train8/train1/weights/best.pt"
ocr_model_path = r"D:/KNN/Cắt ảnh/Cắt ảnh/bestRead.pt"

plate_model = YOLO(plate_model_path)
ocr_model = YOLO(ocr_model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
plate_model.to(device)
ocr_model.to(device)

FRAME_SKIP = 10
CONF_THRESHOLD = 0.3
OCR_CONF_THRESHOLD = 0.4

latest_plates = []
latest_texts = []
lock = threading.Lock()

# -------------------- PHÂN LOẠI BIỂN SỐ --------------------
def classify_plate_shape(plate_img):
    h, w = plate_img.shape[:2]
    aspect_ratio = w / h
    return "horizontal" if aspect_ratio > 2 else "square"

# -------------------- SẮP XẾP KÝ TỰ OCR --------------------
def sort_ocr_results(detected_chars, plate_shape):
    if plate_shape == "horizontal":
        return "".join([char for _, _, char in sorted(detected_chars, key=lambda x: x[0])])
    
    y_values = [y for _, y, _ in detected_chars]
    y_threshold = (max(y_values) + min(y_values)) / 2
    
    top_line = sorted([(x, char) for x, y, char in detected_chars if y < y_threshold], key=lambda x: x[0])
    bottom_line = sorted([(x, char) for x, y, char in detected_chars if y >= y_threshold], key=lambda x: x[0])
    
    return "".join([char for _, char in top_line]) + " " + "".join([char for _, char in bottom_line])

# -------------------- NHẬN DIỆN BIỂN SỐ --------------------
def detect_plates(frame):
    results = plate_model(frame, conf=CONF_THRESHOLD)
    plates, cropped_plates = [], []

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        
        if plate_model.names[cls_id] in ["plate", "BSV", "BSD"] and conf > CONF_THRESHOLD:
            plates.append((x1, y1, x2, y2))
            cropped_plates.append(frame[y1:y2, x1:x2])
    
    return plates, cropped_plates

# -------------------- NHẬN DIỆN KÝ TỰ BIỂN SỐ --------------------
def recognize_text(cropped_plates):
    recognized_texts = []
    
    for plate_img in cropped_plates:
        if plate_img is None or plate_img.size == 0:
            recognized_texts.append("")
            continue
        
        plate_shape = classify_plate_shape(plate_img)
        ocr_results = ocr_model(plate_img)
        detected_chars = [(int(box.xyxy[0][0]), int(box.xyxy[0][1]), ocr_model.names[int(box.cls[0])])
                          for result in ocr_results for box in result.boxes if float(box.conf[0]) > OCR_CONF_THRESHOLD]
        
        recognized_texts.append(sort_ocr_results(detected_chars, plate_shape) if detected_chars else "")
    
    return recognized_texts


# -------------------- XỬ LÝ ẢNH --------------------
def process_image(image_path):
    global latest_plates, latest_texts
    
    image = cv2.imread(image_path)
    if image is None:
        print("Không thể đọc ảnh:", image_path)
        return
    
    plates, cropped_plates = detect_plates(image)
    latest_texts = recognize_text(cropped_plates)
    latest_plates = cropped_plates
    
    # threading.Thread(target=display_plate_window, daemon=True).start()
    
    image_resized = cv2.resize(image, (1280, 720))
    cv2.imshow("License Plate Recognition", image_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# # -------------------- MAIN --------------------
# def main():
#     input_path = r"D:/KNN/Cắt ảnh/Cắt ảnh/test.mp4"
#     if os.path.isfile(input_path):
#         if input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
#             process_video(input_path)
#         elif input_path.lower().endswith(('.jpg', '.jpeg', '.png')):
#             process_image(input_path)
#         else:
#             print("Định dạng file không được hỗ trợ.")
#     else:
#         print("File không tồn tại:", input_path)

# if __name__ == "__main__":
#     main()
