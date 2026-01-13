# BÁO CÁO CHI TIẾT: XÂY DỰNG HỆ THỐNG NHẬN DIỆN CHỮ CÁI VIẾT TAY SỬ DỤNG CNN

**Sinh viên thực hiện:** [Tên của bạn]  
**Môi trường triển khai:** Arch Linux  
**Ngày thực hiện:** 13/01/2025

---

## 1. Giới thiệu đề tài

### 1.1. Đặt vấn đề
Nhận diện chữ viết tay (Handwriting Recognition) là một bài toán kinh điển nhưng đầy thách thức trong lĩnh vực Thị giác máy tính (Computer Vision). Sự đa dạng trong nét chữ của mỗi cá nhân, độ dày nét bút, và vị trí viết khiến cho việc nhận diện chính xác trở nên khó khăn. Các hệ thống OCR truyền thống thường gặp khó khăn với chữ viết tay tự do.

### 1.2. Mục tiêu đề tài
Xây dựng một ứng dụng Desktop hoàn chỉnh trên Linux có khả năng:
*   Cho phép người dùng viết chữ cái in hoa (A-Z) trực tiếp lên màn hình.
*   Sử dụng mạng nơ-ron tích chập (CNN) để nhận diện với độ chính xác cao (>95%).
*   Giải quyết các vấn đề thực tế: nét vẽ mỏng, chữ viết lệch tâm, sai hướng.

### 1.3. Phạm vi
*   **Input:** Ảnh đen trắng từ Canvas kích thước 300x300 px.
*   **Output:** Chữ cái dự đoán (A-Z) và độ tin cậy (Confidence score).
*   **Dataset:** EMNIST Letters (Extended MNIST).

---

## 2. Đặc tả yêu cầu phần mềm

### 2.1. Yêu cầu chức năng (Functional Requirements)
| Chức năng | Mô tả chi tiết |
| :--- | :--- |
| **Vẽ ký tự** | Người dùng vẽ nét đen trên nền trắng. Hỗ trợ nét vẽ mượt (anti-aliasing). |
| **Dự đoán (Predict)** | Gửi ảnh từ Canvas vào Model, trả về kết quả chữ cái có xác suất cao nhất. |
| **Làm sạch (Clear)** | Xóa toàn bộ nội dung Canvas để bắt đầu phiên mới. |
| **Debug Mode** | (Tính năng ẩn) Tự động lưu ảnh đầu vào sau khi tiền xử lý để kiểm tra lỗi xoay/mờ. |

### 2.2. Yêu cầu phi chức năng (Non-functional Requirements)
*   **Hiệu năng:** Thời gian tiền xử lý và dự đoán < 0.5 giây.
*   **Độ chính xác:** Đạt trên 97% trên tập Train và trên 94% trên tập Test.
*   **Tính ổn định:** Không bị crash khi người dùng vẽ quá nhanh hoặc để trống Canvas.

---

## 3. Công nghệ và Công cụ sử dụng

### 3.1. Ngôn ngữ & Thư viện lõi
*   **Python 3.10+:** Ngôn ngữ chủ đạo nhờ hệ sinh thái AI mạnh mẽ.
*   **TensorFlow 2.x / Keras:** Framework Deep Learning để xây dựng và huấn luyện mô hình.
*   **OpenCV (cv2):** Thư viện xử lý ảnh mạnh mẽ, dùng cho các thuật toán Computer Vision (Threshold, Moments, Dilation).
*   **NumPy:** Xử lý ma trận ảnh hiệu năng cao.

### 3.2. Giao diện người dùng (GUI)
*   **Tkinter:** Được chọn vì tính nhẹ, tích hợp sẵn trong Python và hoạt động cực kỳ ổn định trên môi trường Linux (cụ thể là Arch Linux) mà không cần cài đặt phức tạp như Qt.

### 3.3. Dữ liệu (Dataset)
*   **EMNIST Letters:**
    *   Số lượng: 145,600 ảnh.
    *   Kích thước: 28x28 pixel (Grayscale).
    *   Phân lớp: 26 lớp (A-Z).
    *   Đặc điểm: Dữ liệu gốc bị xoay 90 độ và lật ngang (Transposed), cần xử lý kỹ trước khi train.

---

## 4. Phát triển và Triển khai (Chi tiết kỹ thuật & Mã nguồn)

Trong phần này, chúng tôi sẽ đi sâu vào phân tích kiến trúc mô hình và quy trình xử lý ảnh, giải thích lý do lựa chọn từng lớp mạng và thuật toán, kèm theo mã nguồn thực tế.

### 4.1. Kiến trúc Mô hình CNN (Convolutional Neural Network)

Mô hình được xây dựng dựa trên kiến trúc VGG-style (xếp chồng các lớp tích chập) để trích xuất đặc trưng từ đơn giản đến phức tạp.

#### a. Khối trích xuất đặc trưng (Feature Extraction)
Chúng tôi sử dụng 3 khối tích chập liên tiếp. Mỗi khối bao gồm các lớp sau:

1.  **Conv2D (Convolutional Layer):**
    *   *Công dụng:* Sử dụng các bộ lọc (filters) để quét qua ảnh nhằm phát hiện các đặc trưng cục bộ.
        *   Lớp đầu (32 filters): Phát hiện các cạnh, đường nét đơn giản (ngang, dọc, chéo).
        *   Lớp sau (64, 128 filters): Kết hợp các cạnh để nhận diện hình dạng phức tạp hơn (vòng cung, góc nhọn).
    *   *Code:* `Conv2D(32, (3, 3), activation='relu', ...)`
2.  **BatchNormalization:**
    *   *Công dụng:* Chuẩn hóa đầu ra của lớp trước đó về phân phối chuẩn (mean=0, variance=1). Giúp mô hình hội tụ nhanh hơn và ổn định hơn, cho phép sử dụng learning rate lớn hơn.
    *   *Code:* `BatchNormalization()`
3.  **MaxPooling2D:**
    *   *Công dụng:* Giảm kích thước không gian của ảnh (Downsampling) đi một nửa (ví dụ từ 28x28 xuống 14x14). Giúp giảm số lượng tham số tính toán và giữ lại các đặc trưng nổi bật nhất (bất biến với dịch chuyển nhỏ).
    *   *Code:* `MaxPooling2D((2, 2))`
4.  **Dropout:**
    *   *Công dụng:* Ngẫu nhiên "tắt" một tỷ lệ nơ-ron (ví dụ 20%) trong quá trình huấn luyện. Điều này buộc mạng phải học các đặc trưng mạnh mẽ hơn, ngăn chặn việc ghi nhớ dữ liệu (Overfitting).
    *   *Code:* `Dropout(0.2)`

**Mã nguồn triển khai (`train.py`):**
```python
model = Sequential([
    # --- Khối 1: Đặc trưng cấp thấp ---
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.2),

    # --- Khối 2: Đặc trưng cấp trung ---
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    # --- Khối 3: Đặc trưng cấp cao ---
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.4),
    
    # ... (Phần phân loại bên dưới)
])
```

#### b. Khối phân loại (Classification)
Sau khi trích xuất đặc trưng, dữ liệu được duỗi phẳng và đưa vào mạng nơ-ron đầy đủ (Fully Connected).

1.  **Flatten:** Duỗi ma trận 3 chiều thành vector 1 chiều.
2.  **Dense (256 units):** Lớp ẩn để học các tổ hợp phi tuyến tính của các đặc trưng.
3.  **Dense (Output - 26 units):** Lớp đầu ra với hàm kích hoạt **Softmax**.
    *   *Công dụng:* Chuyển đổi các giá trị đầu ra thành xác suất (tổng bằng 1). Lớp nào có xác suất cao nhất chính là chữ cái được dự đoán.

**Mã nguồn triển khai:**
```python
    # ... (Tiếp theo phần trên)
    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(26, activation='softmax') # 26 lớp tương ứng A-Z
```

### 4.2. Pipeline Tiền xử lý ảnh (Image Preprocessing)

Đây là bước quyết định để đồng bộ ảnh vẽ tay của người dùng với dữ liệu EMNIST.

#### a. Chuẩn hóa đầu vào & Cắt vùng quan tâm (ROI)
*   *Vấn đề:* Ảnh vẽ có nền trắng chữ đen, kích thước 300x300, vùng chữ viết có thể nằm lệch.
*   *Giải pháp:* Chuyển sang Grayscale, đảo màu (thành nền đen chữ trắng giống EMNIST), và cắt sát khung chữ (Bounding Box) để loại bỏ khoảng trắng thừa.

**Mã nguồn (`src/preprocess.py`):**
```python
# Chuyển sang ảnh xám và đảo màu
img = image_pil.convert('L')
img_array = np.array(img)
img_array = cv2.bitwise_not(img_array) # Đảo màu: Trắng -> Đen, Đen -> Trắng

# Ngưỡng hóa để loại bỏ nhiễu mờ
_, img_array = cv2.threshold(img_array, 30, 255, cv2.THRESH_BINARY)

# Tìm khung bao quanh (Bounding Box) và cắt
coords = cv2.findNonZero(img_array)
x, y, w, h = cv2.boundingRect(coords)
cropped = img_array[y:y+h, x:x+w]
```

#### b. Căn giữa theo Trọng tâm (Center of Mass)
*   *Công dụng:* Thay vì căn giữa hình học (Geometric Center), ta tính toán trọng tâm pixel (nơi tập trung nhiều nét vẽ nhất) để đặt vào tâm khung hình 28x28. Đây là chuẩn của dataset MNIST/EMNIST, giúp tăng đáng kể độ chính xác với các chữ cái viết lệch như 'L', 'J'.

**Mã nguồn (`src/preprocess.py`):**
```python
# Tính toán trọng tâm (Moments)
M = cv2.moments(resized)
if M["m00"] != 0:
    cX = M["m10"] / M["m00"]
    cY = M["m01"] / M["m00"]
else:
    cX, cY = new_w / 2, new_h / 2

# Tính toán độ dịch chuyển để đưa trọng tâm về tọa độ (14, 14)
x_start = int(round(14 - cX))
y_start = int(round(14 - cY))
```

#### c. Kỹ thuật Dilation & Transpose (Sửa lỗi thực tế)
*   **Dilation (Làm đậm):** Nét vẽ trên Canvas khi thu nhỏ về 28x28 thường bị mảnh hơn so với nét bút lông trong dataset EMNIST. Dùng Dilation để làm dày nét vẽ.
*   **Transpose (Xoay):** Dataset EMNIST gốc lưu trữ ảnh ở dạng xoay 90 độ và lật. Để model (đã học trên dữ liệu này) hiểu được ảnh người dùng vẽ, ta cần xoay ảnh đầu vào theo hướng tương tự.

**Mã nguồn (`src/preprocess.py`):**
```python
# Làm đậm nét (Dilation) với kernel 2x2
kernel = np.ones((2,2), np.uint8)
final_img = cv2.dilate(final_img, kernel, iterations=1)

# Xoay ảnh (Transpose) để khớp với góc nhìn của Model
final_img = np.transpose(final_img)
```

### 4.3. Chiến lược Huấn luyện (Training Strategy)

#### a. Tăng cường dữ liệu (Data Augmentation)
*   *Công dụng:* Tạo ra các biến thể nhân tạo của dữ liệu train (xoay, dịch chuyển, phóng to). Giúp model học được tính tổng quát, nhận diện tốt ngay cả khi người dùng viết chữ nghiêng hoặc nhỏ.

**Mã nguồn (`train.py`):**
```python
datagen = ImageDataGenerator(
    rotation_range=15,       # Xoay ảnh ngẫu nhiên +/- 15 độ
    width_shift_range=0.15,  # Dịch chuyển ngang
    height_shift_range=0.15, # Dịch chuyển dọc
    zoom_range=0.15,         # Phóng to/thu nhỏ
    shear_range=0.1          # Làm nghiêng ảnh
)
```

#### b. Callbacks (Tối ưu hóa quá trình học)
*   **ReduceLROnPlateau:** Tự động giảm Learning Rate khi Loss không giảm, giúp model tìm được điểm cực tiểu tốt hơn (tránh dao động quanh đích).
*   **EarlyStopping:** Dừng huấn luyện sớm nếu model không cải thiện sau một số epoch nhất định, tiết kiệm thời gian và tránh Overfitting.

**Mã nguồn (`train.py`):**
```python
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001)
early_stopping = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
```

---

## 5. Kết luận và Hướng phát triển

### 5.1. Kết quả đạt được
*   **Hệ thống hoàn chỉnh:** Đã xây dựng thành công ứng dụng Desktop nhận diện chữ viết tay trên Linux.
*   **Độ chính xác cao:** Mô hình đạt độ chính xác trên 95% với tập kiểm thử và hoạt động tốt với chữ viết tay thực tế.
*   **Xử lý ảnh tối ưu:** Đã giải quyết triệt để các vấn đề như nét vẽ mảnh, chữ viết lệch tâm, và sai hướng xoay nhờ pipeline tiền xử lý (Center of Mass, Dilation, Transpose).

### 5.2. Hạn chế
*   Chỉ mới hỗ trợ chữ cái in hoa (A-Z), chưa hỗ trợ chữ thường và số.
*   Chưa hỗ trợ viết liền nét (cursive) hoặc nhận diện cả từ/câu.

### 5.3. Hướng phát triển
*   **Mở rộng bộ ký tự:** Sử dụng bộ dữ liệu **EMNIST ByClass** (62 lớp) để nhận diện cả chữ thường (a-z) và chữ số (0-9).
*   **Nhận diện chuỗi:** Tích hợp thuật toán Segmentation (cắt ảnh) để tách và nhận diện từng chữ cái trong một từ hoặc câu hoàn chỉnh.
*   **Đa nền tảng:** Chuyển đổi mô hình sang TensorFlow Lite để phát triển ứng dụng trên thiết bị di động (Android/iOS).
