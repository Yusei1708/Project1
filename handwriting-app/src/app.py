import tkinter as tk
import os
import sys

# Add current directory to path so imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ui import HandwritingApp

def main():
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model', 'letter_cnn_emnist.h5')
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please run 'python train.py' first to generate the model.")
        return

    root = tk.Tk()
    app = HandwritingApp(root, model_path)
    root.mainloop()

if __name__ == "__main__":
    main()
