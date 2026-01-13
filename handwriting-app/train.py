import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import emnist
import cv2  # Thêm vào để lưu ảnh debug

def load_data():
    print("Đang tải bộ dữ liệu EMNIST Letters...")
    # EMNIST Letters: 145,600 ký tự. 26 lớp cân bằng.
    # Nhãn gốc là 1-26 (A-Z). Cần dịch chuyển về 0-25.
    images, labels = emnist.extract_training_samples('letters')
    test_images, test_labels = emnist.extract_test_samples('letters')

    # Ảnh EMNIST mặc định bị xoay 90 độ và lật.
    # Chúng ta cần transpose (xoay) lại để ảnh đứng thẳng giống như người dùng vẽ.
    print("Đang xoay ảnh về đúng hướng...")
    images = np.transpose(images, (0, 2, 1))
    test_images = np.transpose(test_images, (0, 2, 1))

    # --- DEBUG: Lưu mẫu ảnh training để kiểm chứng hướng ảnh ---
    # Lưu vài ảnh đầu tiên ra đĩa để người dùng kiểm tra xem ảnh có đứng thẳng không.
    if not os.path.exists('debug_samples'):
        os.makedirs('debug_samples')
    
    # Lưu 5 mẫu
    for i in range(5):
        # Scale lại về 0-255 để lưu ảnh
        sample_img = images[i].astype(np.uint8)
        cv2.imwrite(f'debug_samples/train_sample_{i}.png', sample_img)
    
    print("DEBUG: Đã lưu các mẫu ảnh training vào thư mục 'debug_samples/'. Vui lòng kiểm tra!")
    # ---------------------------------------------------------------

    # Chuẩn hóa ảnh về khoảng [0, 1]
    images = images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0

    # Reshape về (N, 28, 28, 1) để đưa vào CNN
    images = np.expand_dims(images, axis=-1)
    test_images = np.expand_dims(test_images, axis=-1)

    # Dịch chuyển nhãn từ 1-26 về 0-25
    labels = labels - 1
    test_labels = test_labels - 1

    # One-hot encoding (chuyển nhãn thành vector xác suất)
    labels = to_categorical(labels, num_classes=26)
    test_labels = to_categorical(test_labels, num_classes=26)

    return (images, labels), (test_images, test_labels)

def build_model():
    model = Sequential([
        # Lớp Convolution 1
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.2),

        # Lớp Convolution 2
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        # Lớp Convolution 3
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.4),

        # Các lớp Dense (Fully Connected)
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(26, activation='softmax') # 26 lớp đầu ra (A-Z)
    ])
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train():
    (X_train, y_train), (X_test, y_test) = load_data()
    
    model = build_model()
    model.summary()

    # Tăng cường dữ liệu (Data Augmentation) để model học tốt hơn với các biến thể
    datagen = ImageDataGenerator(
        rotation_range=15,       # Xoay nhẹ
        width_shift_range=0.15,  # Dịch ngang
        height_shift_range=0.15, # Dịch dọc
        zoom_range=0.15,         # Phóng to/thu nhỏ
        shear_range=0.1          # Biến dạng nghiêng
    )
    datagen.fit(X_train)

    # Callbacks: Giảm learning rate và dừng sớm nếu không cải thiện
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1)

    # Bắt đầu huấn luyện
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=64),
        epochs=30,
        validation_data=(X_test, y_test),
        callbacks=[reduce_lr, early_stopping],
        verbose=1
    )

    # Đánh giá model trên tập test
    loss, acc = model.evaluate(X_test, y_test)
    print(f"Độ chính xác trên tập Test: {acc*100:.2f}%")

    # Lưu model
    if not os.path.exists('model'):
        os.makedirs('model')
    model.save('model/letter_cnn_emnist.h5')
    print("Đã lưu model vào model/letter_cnn_emnist.h5")

if __name__ == "__main__":
    train()
