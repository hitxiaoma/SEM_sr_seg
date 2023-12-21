# Time: 2023/12/16 15:05
# Author: Yiming Ma
# Place: Shenzhen
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np

import torch
from torchvision import transforms

import models
from utils import make_coord
from test import batched_predict

class ImageSuperResolutionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Super Resolution App")

        self.image_path = None
        self.model_path = 'C:/Users/28958/Desktop/网络部署/liif-single-software/checkpoint/epoch-200.pth'
        self.resolution = '64,64'

        self.zoom_factor_label = tk.Label(self.master, text="Zoom Factor:")
        self.zoom_factor_label.pack()
        self.zoom_factor_entry = tk.Entry(self.master)
        self.zoom_factor_entry.pack()

        # Create Zoom Button
        self.zoom_button = tk.Button(self.master, text="Zoom", command=self.zoom_image)
        self.zoom_button.pack(pady=10)

        self.create_widgets()
        self.processed_image = None

    def create_widgets(self):
        # Create Open Image Button
        self.open_button = tk.Button(self.master, text="Open Image", command=self.open_image)
        self.open_button.pack(pady=10)

        # Create Zoom Factor Entry
        if not hasattr(self, 'zoom_factor_label'):
            self.zoom_factor_label = tk.Label(self.master, text="Zoom Factor:")
            self.zoom_factor_label.pack()
        if not hasattr(self, 'zoom_factor_entry'):
            self.zoom_factor_entry = tk.Entry(self.master)
            self.zoom_factor_entry.pack()

        # Create Zoom Button
        if not hasattr(self, 'zoom_button'):
            self.zoom_button = tk.Button(self.master, text="Zoom", command=self.zoom_image)
            self.zoom_button.pack(pady=10)

        # Create Save Button
        self.save_button = tk.Button(self.master, text="Save Image", command=self.save_image)
        self.save_button.pack(pady=10)

        # Display Image Canvas
        self.canvas = tk.Canvas(self.master, width=400, height=400)
        self.canvas.pack()

    def open_image(self):
        file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("JPEG files", "*.jpg;*.jpeg")])
        if file_path:
            self.image_path = file_path
            self.display_image()

    def zoom_image(self):
        if self.image_path:
            # Get the zoom factor from the user input
            zoom_factor = self.zoom_factor_entry.get()
            try:
                zoom_factor = float(zoom_factor)
            except ValueError:
                tk.messagebox.showerror("Error", "Invalid zoom factor. Please enter a valid number.")
                return

            img = Image.open(self.image_path).convert('L')
            img = np.array(img)
            img = np.stack((img,) * 3, axis=-1)

            img = transforms.ToTensor()(img)

            model = models.make(torch.load(self.model_path)['model'], load_sd=True).cuda()

            h, w = list(map(int, self.resolution.split(',')))
            h_zoomed, w_zoomed = int(h * zoom_factor), int(w * zoom_factor)
            coord = make_coord((h_zoomed, w_zoomed)).cuda()
            cell = torch.ones_like(coord)
            cell[:, 0] *= 2 / h_zoomed
            cell[:, 1] *= 2 / w_zoomed
            pred = batched_predict(model, ((img - 0.5) / 0.5).cuda().unsqueeze(0),
                                   coord.unsqueeze(0), cell.unsqueeze(0), bsize=30000)[0]
            pred = (pred * 0.5 + 0.5).clamp(0, 1).view(h_zoomed, w_zoomed, 3).permute(2, 0, 1).cpu()

            # Store the processed image for later use
            self.processed_image = pred.cpu().numpy()

            # Display the zoomed image
            self.display_image(self.processed_image)

    def save_image(self):
        if self.processed_image is not None:
            save_path = filedialog.asksaveasfilename(defaultextension=".jpeg",
                                                     filetypes=[("JPEG files", "*.jpg;*.jpeg")])
            if save_path:
                # Get the processed image from the stored array
                img_np = self.processed_image

                # Ensure the data type is 'float32'
                img_np = img_np.astype('float32')

                # Normalize the data range to [0, 1]
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

                # Convert to 'uint8' after adjusting the data range
                img_np = (img_np * 255).astype('uint8')

                # Resize the image to the zoomed size
                h_zoomed, w_zoomed = img_np.shape[1], img_np.shape[2]
                img_np = np.array(Image.fromarray(img_np.transpose(1, 2, 0)).resize((w_zoomed, h_zoomed)))

                # Convert to single-channel
                img_np = img_np.mean(axis=-1, keepdims=True).astype('uint8')

                # Save the single-channel PIL.Image as JPEG
                pil_image = Image.fromarray(img_np.reshape((h_zoomed, w_zoomed)))
                pil_image.save(save_path)

    def display_image(self, img=None):
        if img is None:
            img = Image.open(self.image_path).resize((400, 400))
            img = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
            self.canvas.image = img
        else:
            # If the image has only one channel, repeat it to create three channels
            if img.shape[0] == 1:
                img = np.repeat(img, 3, axis=0)

            # Ensure the data type is 'float32'
            img = img.astype('float32')

            # Normalize the data range to [0, 1]
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)

            # Convert to 'uint8' after adjusting the data range
            img = (img * 255).astype('uint8')

            img = Image.fromarray(img.transpose(1, 2, 0)).resize((400, 400))
            img = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
            self.canvas.image = img
            self.zoomed_image = img


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSuperResolutionApp(root)
    root.mainloop()