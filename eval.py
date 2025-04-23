from ultralytics import YOLO  # Muat model

file_path = 'yolo_digit_detection/exp2/'
# Muat model
model = YOLO(file_path+'weights/best.pt')

# Evaluasi model
metrics = model.val()

# Ambil metrik-metrik
precision = metrics.box.p.mean() * 100
recall = metrics.box.r.mean() * 100
map50 = metrics.box.map50 * 100
map5095 = metrics.box.map * 100

# F1 Score = 2 * (precision * recall) / (precision + recall)
f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0.0

# Approximate accuracy menggunakan rumus (TP) / (TP + FP + FN)
# Karena kita tidak punya TN, ini semacam balanced accuracy pendekatan
approx_accuracy = (precision * recall) / 100  # Nilai ini bukan true accuracy tapi pendekatan

# Menyiapkan hasil evaluasi sebagai string
results = (
    f"Precision         : {precision:.2f}%\n"
    f"Recall            : {recall:.2f}%\n"
    f"F1 Score          : {f1_score:.2f}%\n"
    f"mAP@0.5           : {map50:.2f}%\n"
    f"mAP@0.5:0.95      : {map5095:.2f}%\n"
    f"Approx. Accuracy  : {approx_accuracy:.2f}%\n"
)

# Tentukan path untuk menyimpan file
file_path = file_path + 'evaluation_results.txt'

# Menyimpan hasil evaluasi ke dalam file teks
with open(file_path, 'w') as file:
    file.write(results)

print("Hasil evaluasi disimpan di:", file_path)
