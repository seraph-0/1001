import cv2
import numpy as np
import pytesseract
import re

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  

# Resmi oku
image = cv2.imread('8.jpg', cv2.IMREAD_COLOR)

# Görüntüyü 1280x720 çözünürlüğüne yeniden boyutlandır
resized_image = cv2.resize(image, (1280, 720))

# Gri tonlamalıya dönüştür
gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

# Gaussian filtre uygula
filtered_image = cv2.GaussianBlur(gray, (5, 5), 0)

# Sobel X ve Sobel Y filtrelerini uygula
sobel_x = cv2.Sobel(filtered_image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(filtered_image, cv2.CV_64F, 0, 1, ksize=3)

# Sobel X ve Sobel Y filtrelerinin birleşimini hesapla
sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

# Sobel magnitude görüntüsünü normalize et ve uint8 formatına çevir
sobel_magnitude = np.uint8(255 * sobel_magnitude / np.max(sobel_magnitude))

# Eşik değeri ile beyaz bölgeleri belirle
_, binary_image = cv2.threshold(sobel_magnitude, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Beyaz bölgeleri bulmak için kontur tespiti
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

filled_image = np.zeros_like(binary_image)

cv2.drawContours(filled_image, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
# Büyük konturları (beyaz bölgeleri) maskeleme

kernel = np.ones((41, 7), np.uint8)
erosion = cv2.erode(filled_image, kernel, iterations=1)
dilation = cv2.dilate(erosion, kernel, iterations=1)

contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# En büyük binary nesneyi bul
largest_contour = max(contours, key=cv2.contourArea)

# Yeni bir boş görüntü oluştur
largest_contour_image = np.zeros_like(dilation)

# En büyük binary nesneyi çiz
cv2.drawContours(largest_contour_image, [largest_contour], -1, (255, 255, 255), thickness=cv2.FILLED)

# OR işlemi uygula
masked_image = cv2.bitwise_and(resized_image, resized_image, mask=largest_contour_image)

# En büyük kontur bölgesini kes
x, y, w, h = cv2.boundingRect(largest_contour)
cropped_image = masked_image[y:y+h, x:x+w]

median_image = cv2.GaussianBlur(cropped_image, (5,5),0)

# K-means algoritması ile renk analizi yap
Z = cropped_image.reshape((-1, 3))
Z = np.float32(Z)

# K-means kriterleri ve uygulaması
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
K = 2  # K-means için küme sayısı
_, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Merkezleri uint8'e çevir
centers = np.uint8(centers)
res = centers[labels.flatten()]
result_image = res.reshape((cropped_image.shape))

# Renk kümelerini ve yüzdelerini hesapla
def calculate_color_percentages(labels, centers):
    # Her kümenin yüzdesini hesapla
    label_counts = np.bincount(labels)
    total_count = len(labels)
    percentages = label_counts / total_count * 100
    return percentages

# Renk kümelerini ve yüzdelerini yazdır
def print_color_percentages(centers, percentages):
    for i, (center, percent) in enumerate(zip(centers, percentages)):
        print(f"Küme {i}: Renk {center} - Yoğunluk % {percent:.2f}")

# Yoğunluk yüzdelerini hesapla ve yazdır
percentages = calculate_color_percentages(labels.flatten(), centers)
print_color_percentages(centers, percentages)

text = pytesseract.image_to_string(median_image, config='--psm 10')
# Harf ve sayılar dışında olan karakterleri temizle
cleaned_text = re.sub(r'[^\w\s]', '', text)
# Format kontrolü için düzenli ifade
#pattern = r'^\d{2} [A-Za-z]{1,4} \d{1,4}$'

print ("Tespit edilen Plaka:", cleaned_text.strip())

'''# Formatı kontrol et
if re.fullmatch(pattern, cleaned_text.strip()):
    print("Tespit edilen Plaka:", cleaned_text.strip())
else:
    print("Üzgünüz Plaka tespit edilemedi.")'''


# Sonuçları göster
#cv2.imshow('Res2ult', median_image)
#cv2.imshow('Result', cropped_image)
cv2.imshow('Res22lt', result_image)


cv2.waitKey(0)  # Bir tuşa basılmasını bekler

# Tüm pencereleri kapat
cv2.destroyAllWindows()