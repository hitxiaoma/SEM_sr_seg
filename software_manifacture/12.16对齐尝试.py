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
            zoom_factor_str = self.zoom_factor_entry.get()
            try:
                zoom_factor = float(zoom_factor_str)
            except ValueError:
                tk.messagebox.showerror("Error", "Invalid zoom factor. Please enter a valid number.")
                return

            img = Image.open(self.image_path).convert('L')
            img_original = np.array(img)
            img_original = np.stack((img_original,) * 3, axis=-1)

            img = transforms.ToTensor()(img_original)

            model = models.make(torch.load(self.model_path)['model'], load_sd=True).cuda()

            # Calculate the zoomed resolution based on the original image size and zoom factor
            h_orig, w_orig = img_original.shape[0], img_original.shape[1]
            h_zoomed, w_zoomed = int(h_orig * zoom_factor), int(w_orig * zoom_factor)

            coord = make_coord((h_zoomed, w_zoomed)).cuda()
            cell = torch.ones_like(coord)
            cell[:, 0] *= 2 / h_zoomed
            cell[:, 1] *= 2 / w_zoomed
            pred = batched_predict(model, ((img - 0.5) / 0.5).cuda().unsqueeze(0),
                                   coord.unsqueeze(0), cell.unsqueeze(0), bsize=30000)[0]
            pred = (pred * 0.5 + 0.5).clamp(0, 1).view(3, h_zoomed, w_zoomed).permute(1, 2, 0).cpu()

            # Store the processed image for later use
            self.processed_image = pred.numpy()

            # Convert the processed image to uint8
            self.processed_image = (self.processed_image * 255).astype('uint8')

            # Display the zoomed image without downscaling
            self.display_image(self.processed_image)

    def display_image(self, img=None):
        if img is None:
            img = Image.open(self.image_path).resize((400, 400))
            img = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
            self.canvas.image = img
        else:
            # Resize the image to fit the canvas
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
            self.canvas.image = img

    def save_image(self):
        if self.processed_image is not None:
            save_path = filedialog.asksaveasfilename(defaultextension=".jpeg",
                                                     filetypes=[("JPEG files", "*.jpg;*.jpeg")])
            if save_path:
                # Get the processed image from the stored array
                img_np = self.processed_image

                # Ensure the data type is 'uint8'
                img_np = (img_np * 255).astype('uint8')

                # If the image has only one channel, remove the channel dimension
                if img_np.shape[0] == 1:
                    img_np = img_np[0]

                # Create a single-channel or three-channel PIL.Image based on the shape
                if img_np.ndim == 2:  # Single-channel image
                    mode = 'L'
                elif img_np.ndim == 3:  # Three-channel image
                    mode = 'RGB'
                else:
                    raise ValueError("Unsupported image format")

                # Create a PIL.Image
                pil_image = Image.fromarray(img_np, mode=mode)
                pil_image.save(save_path)


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSuperResolutionApp(root)
    root.mainloop()
