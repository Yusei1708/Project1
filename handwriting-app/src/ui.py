import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw
import numpy as np
from .preprocess import preprocess_image
from .predict import Predictor
import os

class HandwritingApp:
    def __init__(self, root, model_path):
        self.root = root
        self.root.title("Handwriting Recognition (A-Z)")
        
        self.predictor = Predictor(model_path)
        
        # Canvas settings
        self.canvas_width = 300
        self.canvas_height = 300
        self.bg_color = "white"
        self.paint_color = "black"
        
        # TĂNG KÍCH THƯỚC CỌ VẼ (Quan trọng để khi resize không bị mất nét)
        self.brush_size = 25
        
        # UI Layout
        self.create_widgets()
        
        # Drawing state
        self.image = Image.new("RGB", (self.canvas_width, self.canvas_height), self.bg_color)
        self.draw = ImageDraw.Draw(self.image)
        self.last_x, self.last_y = None, None
        
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

    def create_widgets(self):
        # Main container
        main_frame = tk.Frame(self.root, padx=10, pady=10)
        main_frame.pack()
        
        # Canvas
        self.canvas = tk.Canvas(main_frame, width=self.canvas_width, height=self.canvas_height, bg=self.bg_color, relief=tk.RIDGE, bd=2)
        self.canvas.pack(pady=10)
        
        # Buttons Frame
        btn_frame = tk.Frame(main_frame)
        btn_frame.pack(pady=5)
        
        self.btn_clear = tk.Button(btn_frame, text="Clear", command=self.clear_canvas, width=10)
        self.btn_clear.grid(row=0, column=0, padx=5)
        
        self.btn_predict = tk.Button(btn_frame, text="Predict", command=self.predict_digit, width=10, bg="#dddddd")
        self.btn_predict.grid(row=0, column=1, padx=5)
        
        # Result Label
        self.lbl_result = tk.Label(main_frame, text="Draw a letter and click Predict", font=("Helvetica", 14))
        self.lbl_result.pack(pady=10)

    def paint(self, event):
        if self.last_x and self.last_y:
            x, y = event.x, event.y
            self.canvas.create_line((self.last_x, self.last_y, x, y), width=self.brush_size, fill=self.paint_color, capstyle=tk.ROUND, smooth=True)
            self.draw.line((self.last_x, self.last_y, x, y), fill=self.paint_color, width=self.brush_size, joint="curve")
        self.last_x = event.x
        self.last_y = event.y

    def reset(self, event):
        self.last_x, self.last_y = None, None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("RGB", (self.canvas_width, self.canvas_height), self.bg_color)
        self.draw = ImageDraw.Draw(self.image)
        self.lbl_result.config(text="Draw a letter and click Predict")

    def predict_digit(self):
        processed_img = preprocess_image(self.image)
        
        if processed_img is None:
            self.lbl_result.config(text="Canvas is empty!")
            return

        letter, confidence = self.predictor.predict(processed_img)
        
        self.lbl_result.config(text=f"Prediction: {letter} ({confidence*100:.2f}%)")
