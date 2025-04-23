import os
import json
import numpy as np
from ultralytics import YOLO
import easyocr
import cv2
import re
from PIL import Image, ImageEnhance, ImageFilter

# Folder input dan output
input_folder = 'imagedummy'
output_folder = os.path.join(input_folder, 'hasil')
json_output_file = os.path.join(input_folder, 'bib_numbers.json')

# Pastikan folder hasil ada, jika belum buat
os.makedirs(output_folder, exist_ok=True)

# Load YOLO model hasil training
model = YOLO('yolo_digit_detection/exp39/weights/best.pt')

# Load EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False, model_storage_directory='./ocr_models',
                        recog_network='english_g2')  # set gpu=True jika pakai GPU

# Fungsi perbaikan gambar
def apply_image_enhancements(img):
    enhanced_images = []
    enhanced_images.append(("original", img))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enhanced_images.append(("grayscale", gray))
    _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    enhanced_images.append(("thresh_otsu", thresh_otsu))
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)
    enhanced_images.append(("adaptive_thresh", adaptive_thresh))
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    enhanced_images.append(("sharpened", sharpened))
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    enhanced_images.append(("denoised", denoised))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(gray)
    enhanced_images.append(("contrast_enhanced", contrast_enhanced))
    kernel = np.ones((3, 3), np.uint8)
    morph_opened = cv2.morphologyEx(thresh_otsu, cv2.MORPH_OPEN, kernel)
    enhanced_images.append(("morph_opened", morph_opened))
    return enhanced_images

def apply_pil_enhancements(img):
    pil_enhancements = []
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    pil_enhancements.append(("pil_original", pil_img))
    enhancer = ImageEnhance.Sharpness(pil_img)
    sharp_img = enhancer.enhance(2.0)
    pil_enhancements.append(("pil_sharp", sharp_img))
    enhancer = ImageEnhance.Contrast(pil_img)
    contrast_img = enhancer.enhance(2.0)
    pil_enhancements.append(("pil_contrast", contrast_img))
    unsharp_img = pil_img.filter(ImageFilter.UnsharpMask(radius=2, percent=150))
    pil_enhancements.append(("pil_unsharp", unsharp_img))
    gray_img = pil_img.convert('L')
    threshold_img = gray_img.point(lambda x: 0 if x < 128 else 255, '1')
    pil_enhancements.append(("pil_threshold", threshold_img))
    return pil_enhancements

def validate_bib_number(text, min_chars=1, max_chars=6):
    cleaned_text = re.sub(r'\D', '', text)
    if len(cleaned_text) >= min_chars and len(cleaned_text) <= max_chars:
        return cleaned_text
    return None

def process_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ Gambar tidak dapat dibaca: {img_path}")
        return None

    results = model(img)
    detected_bib_numbers = []
    best_results_per_bbox = {}

    for r in results:
        for i, box in enumerate(r.boxes):
            yolo_conf = float(box.conf[0])  # Confidence YOLO
            if yolo_conf < 0.80:  # Filter YOLO confidence rendah
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bbox_id = f"{x1}_{y1}_{x2}_{y2}"
            cropped = img[y1:y2, x1:x2]

            try:
                ocr_subresults = reader.readtext(cropped, allowlist='0123456789')

                combined_digits = ""
                total_confidence = 0
                count_valid = 0

                for detection in ocr_subresults:
                    _, text, conf = detection
                    clean_digit = validate_bib_number(text)
                    if clean_digit:
                        combined_digits += clean_digit
                        total_confidence += conf
                        count_valid += 1

                if 1 <= len(combined_digits) <= 6 and count_valid > 0:
                    avg_conf = total_confidence / count_valid
                    if avg_conf >= 0.80:
                        best_results_per_bbox[bbox_id] = (combined_digits, avg_conf, yolo_conf)

                        output_crop_filename = os.path.join(output_folder, f"crop_bib_{combined_digits}_{bbox_id}.jpg")
                        cv2.imwrite(output_crop_filename, cropped)

            except Exception as e:
                print(f"⚠️ Error OCR pada bounding box: {e}")


    img_with_boxes = img.copy()
    for bbox_id, (bib_number, ocr_confidence, yolo_confidence) in best_results_per_bbox.items():
        if ocr_confidence >= 0.80 and yolo_confidence >= 0.80:
            x1, y1, x2, y2 = map(int, bbox_id.split('_'))
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_with_boxes, f"{bib_number} ({ocr_confidence:.2f}/{yolo_confidence:.2f})",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            detected_bib_numbers.append({
                'bbox_id': bbox_id,
                'bib_number': bib_number,
                'ocr_confidence': round(ocr_confidence, 2),
                'yolo_confidence': round(yolo_confidence, 2),
                'image_path': img_path
            })


    hasilbib_folder = os.path.join(input_folder, 'hasilbib')
    os.makedirs(hasilbib_folder, exist_ok=True)
    output_img_path = os.path.join(hasilbib_folder, os.path.basename(img_path))
    cv2.imwrite(output_img_path, img_with_boxes)

    return detected_bib_numbers

def process_images_in_folder(input_folder):
    bib_numbers = []
    for filename in sorted(os.listdir(input_folder)):  # Sorting filenames alphabetically
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            print(f"Memproses {filename}...")
            img_path = os.path.join(input_folder, filename)
            detected_bib_numbers = process_image(img_path)
            if detected_bib_numbers:
                bib_numbers.extend(detected_bib_numbers)
    return bib_numbers

# Main process
detected_bib_numbers = process_images_in_folder(input_folder)

# Simpan hasil ke file JSON
with open(json_output_file, 'w') as json_file:
    json.dump(detected_bib_numbers, json_file, indent=4)

print(f"Proses selesai. Hasil disimpan di {json_output_file}")
