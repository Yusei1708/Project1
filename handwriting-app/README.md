# Handwriting Recognition App

This application recognizes handwritten letters (A-Z) using a CNN trained on the EMNIST Letters dataset.

## Setup

1.  Install dependencies:
    ```bash
    pip install tensorflow numpy opencv-python pillow matplotlib emnist
    ```

2.  Train the model (required first time):
    ```bash
    python train.py
    ```
    This will generate `model/letter_cnn_emnist.h5`.

3.  Run the application:
    ```bash
    python src/app.py
    ```

## Structure

*   `model/`: Contains the trained model.
*   `src/`: Source code for the application.
    *   `app.py`: Entry point.
    *   `ui.py`: User interface.
    *   `predict.py`: Prediction logic.
    *   `preprocess.py`: Image preprocessing.
*   `train.py`: Script to train the CNN model.
