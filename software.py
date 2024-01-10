import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import torch
from torchvision import transforms
import os
from mmseg.apis1 import init_segmentor, inference_segmentor
import models
from utils import make_coord
from test import batched_predict
import matplotlib.pyplot as plt
import pandas as pd


class ImageSuperResolutionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("混凝土SEM图像修复与分割软件V1.0")

        icon_path = "logo.ico"
        self.master.iconbitmap(icon_path)

        self.image_path = None
        self.model_path = os.path.join(os.path.dirname(__file__), 'checkpoint/LIIF.pth')
        self.seg_path = os.path.join(os.path.dirname(__file__), 'configs/segformer.b0.512x512.ade.160k.py')
        self.seg_ckp_path = os.path.join(os.path.dirname(__file__), 'checkpoint/Segformer.pth')
        self.zoom_factor_entry = None

        self.create_widgets()
        self.processed_image = None
        self.seg_image = None
        self.sum_pixel_image = None
        self.pixel_proportion = None
        self.master.minsize(width=600, height=400)

    def create_widgets(self):

        button_frame = tk.Frame(self.master)
        button_frame.pack(side=tk.TOP)

        self.open_button = tk.Button(button_frame, text="打开图像", command=self.open_image)
        self.open_button.pack(side=tk.LEFT, padx=10)

        self.crop_button = tk.Button(button_frame, text="裁剪底部像素点", command=self.crop_image)
        self.crop_button.pack(side=tk.LEFT, padx=10)

        self.zoom_button = tk.Button(button_frame, text="超分辨率重构", command=self.zoom_image)
        self.zoom_button.pack(side=tk.LEFT, padx=10)

        self.segmentation_button = tk.Button(button_frame, text="物相分割", command=self.segment_image)
        self.segmentation_button.pack(side=tk.LEFT, padx=10)

        self.save_button = tk.Button(button_frame, text="保存图片", command=self.save_image)
        self.save_button.pack(side=tk.LEFT, padx=10)

        self.proportions_button = tk.Button(button_frame, text="分割后物相比例", command=self.proportions)
        self.proportions_button.pack(side=tk.LEFT, padx=10)

        self.save_proportions_button = tk.Button(button_frame, text="保存物相比例", command=self.save_proportions_data,
                                                 state=tk.DISABLED)
        self.save_proportions_button.pack(side=tk.LEFT, padx=10)

        self.canvas = tk.Canvas(self.master, width=600, height=400)
        self.canvas.pack(side=tk.TOP, pady=10, expand=True, fill=tk.BOTH)

    def open_image(self):
        file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("JPEG files", "*.jpg;*.jpeg")])
        if file_path:
            self.image_path = file_path
            self.display_image()

    def crop_image(self):
        if self.image_path:
            try:
                crop_pixels = int(simpledialog.askstring("Input", "请选择需要裁剪的行数："))
            except ValueError:
                messagebox.showerror("Error", "Invalid input. Please enter a valid number.")
                return

            img = Image.open(self.image_path).convert('L')
            img = np.array(img)
            img = img[:-crop_pixels, :]

            self.cropped_image_path = f"cropped_{crop_pixels}px_{os.path.basename(self.image_path)}"
            Image.fromarray(img).save(self.cropped_image_path)
            img = np.expand_dims(img, axis=0)
            self.display_image(img)

    def zoom_image(self):

        if hasattr(self, 'cropped_image_path'):
            try:
                zoom_factor = int(simpledialog.askstring("Input", "请选择超分辨率重构倍数："))
            except ValueError:
                messagebox.showerror("Error", "Invalid input. Please enter a valid number.")
                return

            img = Image.open(self.cropped_image_path).convert('L')
            img = np.array(img)
            img = np.stack((img,) * 3, axis=-1)

            img = transforms.ToTensor()(img)

            model = models.make(torch.load(self.model_path)['model'], load_sd=True).cuda()

            h_orig, w_orig = img.shape[1], img.shape[2]
            h_zoomed, w_zoomed = int(h_orig * zoom_factor), int(w_orig * zoom_factor)

            coord = make_coord((h_zoomed, w_zoomed)).cuda()
            cell = torch.ones_like(coord)
            cell[:, 0] *= 2 / h_zoomed
            cell[:, 1] *= 2 / w_zoomed
            pred = batched_predict(model, ((img - 0.5) / 0.5).cuda().unsqueeze(0),
                                   coord.unsqueeze(0), cell.unsqueeze(0), bsize=30000)[0]
            pred = (pred * 0.5 + 0.5).clamp(0, 1).view(h_zoomed, w_zoomed, 3).permute(2, 0, 1).cpu()

            self.processed_image = pred.cpu().numpy()
            self.display_image(self.processed_image)

    def segment_image(self):
        if self.processed_image is not None:
            segmented_result = self.perform_segmentation(self.processed_image)

            img_array = np.array(segmented_result)
            segmented_result_expanded = np.expand_dims(img_array, axis=0)

            segmented_result_3d = np.repeat(segmented_result_expanded, 3, axis=0)

            self.seg_image = segmented_result_3d
            self.display_image(segmented_result_3d)

    def perform_segmentation(self, img):
        temp_path = "temp_processed_image.jpeg"
        processed_image_uint8 = (self.processed_image * 255).astype(np.uint8)
        processed_image_pil = Image.fromarray(np.transpose(processed_image_uint8, (1, 2, 0)))
        processed_image_pil.save(temp_path, format='JPEG')
        image = Image.open(temp_path)

        model = init_segmentor(self.seg_path, self.seg_ckp_path, device='cuda:0')
        result = inference_segmentor(model, temp_path)[0]
        self.sum_pixel_image = result
        result = Image.fromarray(np.uint8(result * 100))

        return result

    def proportions(self):
        if self.sum_pixel_image is not None:
            result_integer = self.sum_pixel_image.astype(int)

            unique, counts = np.unique(result_integer, return_counts=True)
            pixel_value_counts = {0: 0, 1: 0, 2: 0}
            for value, count in zip(unique, counts):
                pixel_value_counts[value] += count

            total_pixels = sum(pixel_value_counts.values())
            proportions = [pixel_value_counts[key] / total_pixels for key in sorted(pixel_value_counts.keys())]

            self.pixel_proportion = proportions

            plt.rcParams['font.family'] = 'SimHei'
            plt.rcParams['font.size'] = 10
            labels = {0: '孔洞', 1: '水化产物', 2: '未水化水泥颗粒'}

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

            ax1.imshow(self.sum_pixel_image, cmap='viridis')
            ax1.axis('off')
            ax1.set_title('物相分割图')

            ax2.pie(proportions, labels=[labels[key] for key in sorted(labels.keys())], autopct='%1.1f%%', startangle=140)
            ax2.set_title('物相分割比例')

            temp_chart_path = "temp_chart.png"
            fig.savefig(temp_chart_path)

            if hasattr(self, 'chart_label'):
                self.chart_label.destroy()

            chart_image = Image.open(temp_chart_path)
            chart_tk_image = ImageTk.PhotoImage(chart_image)
            self.chart_label = tk.Label(self.master, image=chart_tk_image)
            self.chart_label.image = chart_tk_image
            self.chart_label.pack()
            self.save_proportions_button.config(state=tk.NORMAL)
            os.remove(temp_chart_path)

            return proportions

    def save_proportions_data(self):
        proportions = self.pixel_proportion
        save_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if save_path:
            data = pd.DataFrame({"物相": ["孔洞", "水化产物", "未水化水泥颗粒"], "比例": proportions})
            data.to_csv(save_path, index=False, sep='\t')
            messagebox.showinfo("保存成功", f"物相比例数据已保存至：\n{save_path}")

    def save_image(self):

        if self.seg_image is not None:
            save_path = filedialog.asksaveasfilename(defaultextension=".jpeg",
                                                     filetypes=[("JPEG files", "*.jpg;*.jpeg")])
            if save_path:
                img_np = self.seg_image
                img_np = img_np.astype('float32')
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
                img_np = (img_np * 255).astype('uint8')

                h_zoomed, w_zoomed = img_np.shape[1], img_np.shape[2]
                img_np = np.array(Image.fromarray(img_np.transpose(1, 2, 0)).resize((w_zoomed, h_zoomed)))
                img_np = img_np.mean(axis=-1, keepdims=True).astype('uint8')
                pil_image = Image.fromarray(img_np.reshape((h_zoomed, w_zoomed)))
                pil_image.save(save_path)
        else:
            if self.processed_image is not None:
                save_path = filedialog.asksaveasfilename(defaultextension=".jpeg",
                                                         filetypes=[("JPEG files", "*.jpg;*.jpeg")])
                if save_path:
                    img_np = self.processed_image
                    img_np = img_np.astype('float32')
                    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
                    img_np = (img_np * 255).astype('uint8')

                    h_zoomed, w_zoomed = img_np.shape[1], img_np.shape[2]
                    img_np = np.array(Image.fromarray(img_np.transpose(1, 2, 0)).resize((w_zoomed, h_zoomed)))
                    img_np = img_np.mean(axis=-1, keepdims=True).astype('uint8')

                    pil_image = Image.fromarray(img_np.reshape((h_zoomed, w_zoomed)))
                    pil_image.save(save_path)

    def display_image(self, img=None):
        if img is None:
            img_path = self.image_path
            original_img = Image.open(img_path).convert('L')

            aspect_ratio = original_img.width / original_img.height
            new_height = int(400 / aspect_ratio)
            resized_img = original_img.resize((400, new_height))

            img = ImageTk.PhotoImage(resized_img)
            self.canvas.create_image(140, 50, anchor=tk.NW, image=img)
            self.canvas.image = img
        else:
            if img.shape[0] == 1:
                img = np.repeat(img, 3, axis=0)
            img = img.astype('float32')
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img = (img * 255).astype('uint8')
            img = Image.fromarray(img.transpose((1, 2, 0)))

            aspect_ratio = img.width / img.height
            resized_img = img.resize((400, int(400 / aspect_ratio)))
            img = ImageTk.PhotoImage(resized_img)
            self.canvas.create_image(140, 50, anchor=tk.NW, image=img)
            self.canvas.image = img
            self.zoomed_image = img


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSuperResolutionApp(root)
    root.mainloop()