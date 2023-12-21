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
        # self.resolution = '64,64'

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

            # Calculate new dimensions based on zoom factor
            h, w = img.shape
            h_zoomed, w_zoomed = int(h * zoom_factor), int(w * zoom_factor)

            # Prepare coordinates for each chunk
            coord = make_coord((h_zoomed, w_zoomed)).cuda()
            cell = torch.ones_like(coord)
            cell[:, 0] *= 2 / h_zoomed
            cell[:, 1] *= 2 / w_zoomed

            # Process the image in chunks
            chunk_size = 64
            output_chunks = []

            for i in range(0, h, chunk_size):
                for j in range(0, w, chunk_size):
                    chunk = img[i:i + chunk_size, j:j + chunk_size]

                    # If the chunk is smaller than 64x64, pad it
                    if chunk.shape[0] < chunk_size or chunk.shape[1] < chunk_size:
                        pad_h = chunk_size - chunk.shape[0] if chunk.shape[0] < chunk_size else 0
                        pad_w = chunk_size - chunk.shape[1] if chunk.shape[1] < chunk_size else 0
                        chunk = np.pad(chunk, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

                    chunk = np.stack((chunk,) * 3, axis=-1)
                    chunk = transforms.ToTensor()(chunk)

                    model = models.make(torch.load(self.model_path)['model'], load_sd=True).cuda()
                    pred = batched_predict(model, ((chunk - 0.5) / 0.5).cuda().unsqueeze(0),
                                           coord.unsqueeze(0), cell.unsqueeze(0), bsize=30000)[0]

                    pred = (pred * 0.5 + 0.5).clamp(0, 1).view(h_zoomed, w_zoomed, 3).permute(2, 0, 1).cpu()
                    output_chunks.append(pred.numpy())

            # Reshape and stitch the chunks together
            output_image = np.concatenate([np.concatenate(output_chunks[:2], axis=2),
                                           np.concatenate(output_chunks[2:], axis=2)], axis=1).transpose(1, 2, 0)

            # Resize the image to the target size (256x256)
            output_image = Image.fromarray((output_image * 255).astype('uint8'))
            output_image = output_image.resize((h_zoomed, w_zoomed), Image.BICUBIC)
            output_image = np.array(output_image)

            # Store the processed image for later use
            self.processed_image = output_image

            # Display the zoomed image
            self.display_image(self.processed_image)

    def save_image(self):
        if self.processed_image is not None:
            save_path = filedialog.asksaveasfilename(defaultextension=".jpeg",
                                                     filetypes=[("JPEG files", "*.jpg;*.jpeg")])
            if save_path:
                # Get the processed image from the stored array
                img_np = self.processed_image

                # Ensure the processed image has the correct shape
                if len(img_np.shape) == 2:
                    img_np = np.repeat(img_np[:, :, np.newaxis], 3, axis=2)

                # Ensure the data type is 'float32'
                img_np = img_np.astype('float32')

                # Normalize the data range to [0, 1]
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

                # Convert to 'uint8' after adjusting the data range
                img_np = (img_np * 255).astype('uint8')

                # Create a PIL Image from the processed image
                pil_image = Image.fromarray(img_np)

                # Save the PIL Image as JPEG
                pil_image.save(save_path)

    def display_image(self, img=None):
        if img is None:
            img = Image.open(self.image_path).resize((400, 400))
            img = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
            self.canvas.image = img
        else:
            # Ensure the processed image has the correct shape
            if len(img.shape) == 2:
                img = np.repeat(img[:, :, np.newaxis], 3, axis=2)

            # Ensure the data type is 'float32'
            img = img.astype('float32')

            # Normalize the data range to [0, 1]
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)

            # Convert to 'uint8' after adjusting the data range
            img = (img * 255).astype('uint8')

            # Resize the image to fit the canvas
            h_canvas, w_canvas = 400, 400
            img = Image.fromarray(img).resize((w_canvas, h_canvas))
            img = ImageTk.PhotoImage(img)

            self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
            self.canvas.image = img
            self.zoomed_image = img


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSuperResolutionApp(root)
    root.mainloop()
