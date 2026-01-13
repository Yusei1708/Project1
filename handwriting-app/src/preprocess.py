import cv2
import numpy as np
from PIL import Image
import math

def preprocess_image(image_pil):
    """
    Tiền xử lý ảnh từ Canvas để khớp với định dạng EMNIST.
    """
    # Chuyển sang ảnh xám (Grayscale)
    img = image_pil.convert('L')
    img_array = np.array(img)

    # Đảo ngược màu (Canvas: nền trắng chữ đen -> Model: nền đen chữ trắng)
    img_array = cv2.bitwise_not(img_array)

    # Ngưỡng hóa (Threshold) để loại bỏ nhiễu
    _, img_array = cv2.threshold(img_array, 30, 255, cv2.THRESH_BINARY)

    # Tìm khung bao quanh chữ (Bounding Box)
    coords = cv2.findNonZero(img_array)
    if coords is None:
        return None # Canvas trống
    
    x, y, w, h = cv2.boundingRect(coords)
    
    # Cắt lấy phần chữ
    cropped = img_array[y:y+h, x:x+w]
    
    # Resize về kích thước chuẩn (giữ tỷ lệ khung hình)
    # Mục tiêu: Chữ nằm gọn trong hộp 20x20
    target_size = 20
    h_c, w_c = cropped.shape
    
    scale = target_size / max(h_c, w_c)
    new_h, new_w = int(h_c * scale), int(w_c * scale)
    
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Tạo ảnh nền đen 28x28
    final_img = np.zeros((28, 28), dtype=np.uint8)
    
    # Căn giữa theo trọng tâm (Center of Mass)
    M = cv2.moments(resized)
    if M["m00"] != 0:
        cX = M["m10"] / M["m00"]
        cY = M["m01"] / M["m00"]
    else:
        cX, cY = new_w / 2, new_h / 2

    # Chúng ta muốn trọng tâm (cX, cY) nằm tại tâm ảnh 28x28 (tức là tọa độ 14, 14)
    x_start = int(round(14 - cX))
    y_start = int(round(14 - cY))
    
    # Dán ảnh đã resize vào ảnh nền 28x28
    dst_x_start = max(0, x_start)
    dst_y_start = max(0, y_start)
    dst_x_end = min(28, x_start + new_w)
    dst_y_end = min(28, y_start + new_h)
    
    src_x_start = max(0, -x_start)
    src_y_start = max(0, -y_start)
    src_x_end = src_x_start + (dst_x_end - dst_x_start)
    src_y_end = src_y_start + (dst_y_end - dst_y_start)
    
    if (dst_x_end > dst_x_start) and (dst_y_end > dst_y_start):
        final_img[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = resized[src_y_start:src_y_end, src_x_start:src_x_end]
    
    # Làm đậm nét (Dilation)
    kernel = np.ones((2,2), np.uint8)
    final_img = cv2.dilate(final_img, kernel, iterations=1)

    # --- SỬA LỖI XOAY HÌNH ---
    # Vì Model hiện tại của bạn đang học trên dữ liệu bị xoay (Raw EMNIST),
    # nên ta phải xoay ảnh đầu vào để khớp với Model.
    # Transpose tương đương với việc xoay 90 độ và lật.
    final_img = np.transpose(final_img)

    # --- DEBUG: Lưu ảnh để kiểm tra ---
    # Lưu ảnh SAU KHI XOAY để xem nó có khớp với trí tưởng tượng của Model không
    # (Lưu ý: Khi mở ảnh này lên, bạn sẽ thấy nó nằm ngang hoặc lật ngược, 
    # nhưng đó chính là thứ Model muốn thấy).
    cv2.imwrite("debug_input.png", final_img)
    
    # Chuẩn hóa về [0, 1]
    final_img = final_img.astype('float32') / 255.0
    
    # Reshape để đưa vào model
    final_img = np.expand_dims(final_img, axis=0)
    final_img = np.expand_dims(final_img, axis=-1)
    
    return final_img
