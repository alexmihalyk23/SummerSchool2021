import pyvirtualcam
from tkinter import *
import mediapipe as mp
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import cv2
import tkinter.filedialog as tkFileDialog

mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation

path = None


def select_image():
    global path
    path = tkFileDialog.askopenfilename()


def meet(image):
    with mp_selfie_segmentation.SelfieSegmentation(
            model_selection=1) as selfie_segmentation:
        bg_image = None
        if path is not None:
            bg_image = cv2.resize(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB), (640, 480))
        else:
            bg_image = cv2.GaussianBlur(image, (25, 25), 0)
        results = selfie_segmentation.process(image)
        condition = np.stack(
            (results.segmentation_mask,) * 3, axis=-1) > 0.1

        if bg_image is None:
            bg_image = np.zeros(image.shape, dtype=np.uint8)
            bg_image[:] = (192, 192, 192)
        output_image = np.where(condition, image, bg_image)
        return output_image


class MainWindow():
    def __init__(self, window, cap):
        self.window = window
        self.cap = cap
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.interval = 20  # Interval in ms to get the latest frame
        # Create canvas for image
        self.canvas = tk.Canvas(self.window, width=self.width, height=self.height)
        self.canvas.grid(row=0, column=0)
        self.btn = Button(self.window, text="Select an image", command=select_image)
        self.btn.grid(row=1, column=0)
        with pyvirtualcam.Camera(width=640, height=480, fps=20) as cam:
            while True:
                self.update_image(cam)

    def update_image(self,cam):
        # Get the latest frame and convert image format

        # self.image = Image.fromarray(self.image)  # to PIL format
        # self.image = ImageTk.PhotoImage(self.image)  # to ImageTk format
        # Update image
        # self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image)
        # Repeat every 'interval' ms


        self.image = cv2.cvtColor(self.cap.read()[1], cv2.COLOR_BGR2RGB)  # to RGB

        self.image = meet(self.image)

        # frame = cv2.cvtColor(, cv2.COLOR_BGR2RGB)
        cv2.waitKey(0)
        cam.send(cv2.resize(self.image, (640, 480)))
        cam.sleep_until_next_frame()



if __name__ == "__main__":
    root = tk.Tk()
    MainWindow(root, cv2.VideoCapture(0))
    root.mainloop()
