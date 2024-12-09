import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
from yolov7.seg.segment.custom_predict import run_inference, load_model
import json

def annotate_image(image, detections):
    if detections is None or len(detections) == 0:
        return image

    for det in detections:
        bbox = det[:4].tolist()
        confidence = det[4].item()
        cls = int(det[5].item())

        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"Class: {cls} Conf: {confidence:.2f}"
        label_position = (x1, y1 - 10 if y1 - 10 > 10 else y1 + 10)
        cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return image


def annotate_images_batch(results):
    annotated_images = []

    for path, det, _, im0s in results:
        annotated_img = annotate_image(im0s, det)
        annotated_images.append((path, annotated_img))

    return annotated_images

class YOLOv7App:
    CONFIG_FILE = "config.json"

    def __init__(self, root, model):
        self.root = root
        self.model = model
        self.root.title("YOLOv7 Image Processor")
        self.root.geometry("1000x700")

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True)

        self.main_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.main_frame, text="Main")

        self.settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.settings_frame, text="Settings")

        self.gallery_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.gallery_frame, text="Gallery")

        self.process_button = tk.Button(self.main_frame, text="Process Images", command=self.process_folder, state=tk.DISABLED)
        self.process_button.pack(pady=20)

        self.progress_label = tk.Label(self.main_frame, text="Progress:")
        self.progress_label.pack(pady=5)

        self.progress_bar = ttk.Progressbar(self.main_frame, length=900, mode="determinate")
        self.progress_bar.pack(fill="x", padx=10, pady=10)

        canvas_frame = ttk.Frame(self.main_frame)
        canvas_frame.pack(fill="both", expand=True, padx=10, pady=10)

        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.columnconfigure(1, weight=1)
        canvas_frame.rowconfigure(0, weight=1)

        self.before_canvas = tk.Canvas(canvas_frame, bg="gray")
        self.before_canvas.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.after_canvas = tk.Canvas(canvas_frame, bg="gray")
        self.after_canvas.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        self.input_folder_label = tk.Label(self.settings_frame, text="Input Folder:")
        self.input_folder_label.grid(row=0, column=0, padx=10, pady=5)
        self.input_folder_button = tk.Button(self.settings_frame, text="Select Input Folder", command=self.select_input_folder)
        self.input_folder_button.grid(row=0, column=1, padx=10, pady=5)

        self.output_folder_label = tk.Label(self.settings_frame, text="Output Folder:")
        self.output_folder_label.grid(row=1, column=0, padx=10, pady=5)
        self.output_folder_button = tk.Button(self.settings_frame, text="Select Output Folder", command=self.select_output_folder)
        self.output_folder_button.grid(row=1, column=1, padx=10, pady=5)

        self.image_paths = []
        self.annotated_images = []
        self.filtered_images = []
        self.current_image_index = 0
        self.input_folder = ""
        self.output_folder = ""
        self.load_config()

        self.root.bind("<Configure>", self.on_resize)
        
        self.filter_frame = ttk.Frame(self.gallery_frame)
        self.filter_frame.pack(fill="x", padx=10, pady=5)

        self.filter_all_button = tk.Button(self.filter_frame, text="Show All", command=self.show_all_images)
        self.filter_all_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        self.filter_with_detections_button = tk.Button(self.filter_frame, text="Detections", command=self.filter_with_detections)
        self.filter_with_detections_button.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        self.filter_no_detections_button = tk.Button(self.filter_frame, text="No Detections", command=self.filter_no_detections)
        self.filter_no_detections_button.grid(row=0, column=2, padx=5, pady=5, sticky="ew")

        self.filter_frame.grid_columnconfigure(0, weight=1)
        self.filter_frame.grid_columnconfigure(1, weight=1)
        self.filter_frame.grid_columnconfigure(2, weight=1)

        self.gallery_canvas = tk.Canvas(self.gallery_frame)
        self.gallery_canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        self.gallery_scrollbar = ttk.Scrollbar(self.gallery_frame, orient="vertical", command=self.gallery_canvas.yview)
        self.gallery_scrollbar.pack(side="right", fill="y")
        self.gallery_canvas.configure(yscrollcommand=self.gallery_scrollbar.set)

        self.gallery_container = ttk.Frame(self.gallery_canvas)
        self.gallery_canvas.create_window((0, 0), window=self.gallery_container, anchor="nw")
        self.gallery_container.bind("<Configure>", lambda e: self.gallery_canvas.configure(scrollregion=self.gallery_canvas.bbox("all")))

        self.gallery_buttons = []
        self.state = 1
        self.calculate_images_per_row()
        print(self.images_per_row)
        self.check_existing_images()

    def check_existing_images(self):
        """ Check the input and output folders for existing images and display them in the gallery. """
        if not self.input_folder or not self.output_folder:
            print("Folders not selected yet.")
            return

        input_images = [f for f in os.listdir(self.input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        detections_folder = os.path.join(self.output_folder, "detections")
        no_detections_folder = os.path.join(self.output_folder, "no_detections")

        detection_images = [f for f in os.listdir(detections_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
        no_detection_images = [f for f in os.listdir(no_detections_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

        all_output_images = detection_images + no_detection_images

        self.annotated_images = []

        for input_image in input_images:
            input_path = os.path.join(self.input_folder, input_image)
            
            if input_image in all_output_images:
                if input_image in detection_images:
                    output_path = os.path.join(detections_folder, input_image)
                else:
                    output_path = os.path.join(no_detections_folder, input_image)

                if os.path.exists(output_path):
                    annotated_img = cv2.imread(output_path)
                    self.annotated_images.append((input_path, annotated_img))

        self.filtered_images = self.annotated_images
        self.show_gallery_images()

    def show_gallery_images(self):
        """Display the filtered images in the gallery."""
        if self.filtered_images:
            self.images = self.filtered_images
        else:
            print("No images to display in the gallery.")
            self.images = self.annotated_images

        for widget in self.gallery_container.winfo_children():
            widget.destroy()

        row = 0
        col = 0

        for idx, (path, annotated_img) in enumerate(self.images):
            image_frame = ttk.Frame(self.gallery_container)
            image_frame.grid(row=row, column=col, padx=10, pady=10)

            output_img = Image.fromarray(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
            output_img.thumbnail((150, 150))
            output_img_tk = ImageTk.PhotoImage(output_img)
            output_label = tk.Label(image_frame, image=output_img_tk)
            output_label.image = output_img_tk
            output_label.pack()

            button = tk.Button(image_frame, text="Open", command=lambda ip=path: self.explorer_view(ip))
            button.pack()

            col += 1
            if col >= self.images_per_row:
                col = 0
                row += 1

    def calculate_images_per_row(self):
        """Calculate the number of images that can fit in one row based on container width."""
        self.state += 1
        if self.state % 2 == 0:
            self.images_per_row = 5
        else: self.images_per_row = 11
        print(self.images_per_row)
        return

    def show_all_images(self):
        """Display all images in the gallery."""
        self.filtered_images = self.annotated_images
        self.show_gallery_images()

    def filter_with_detections(self):
        """Filter and display images with detections."""
        detections_folder = os.path.join(self.output_folder, "detections")
        detection_images = {f for f in os.listdir(detections_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))}
        
        self.filtered_images = [
            (input_path, annotated_img)
            for input_path, annotated_img in self.annotated_images
            if os.path.basename(input_path) in detection_images
        ]
        self.show_gallery_images()

    def filter_no_detections(self):
        """Filter and display images without detections."""
        no_detections_folder = os.path.join(self.output_folder, "no_detections")
        no_detection_images = {f for f in os.listdir(no_detections_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))}
        
        self.filtered_images = [
            (input_path, annotated_img)
            for input_path, annotated_img in self.annotated_images
            if os.path.basename(input_path) in no_detection_images
        ]
        self.show_gallery_images()

    def save_config(self):
        config_data = {
            "input_folder": self.input_folder,
            "output_folder": self.output_folder
        }
        with open(self.CONFIG_FILE, "w") as config_file:
            json.dump(config_data, config_file)

    def load_config(self):
        if os.path.exists(self.CONFIG_FILE):
            with open(self.CONFIG_FILE, "r") as config_file:
                config_data = json.load(config_file)
                self.input_folder = config_data.get("input_folder", "")
                self.output_folder = config_data.get("output_folder", "")

                if os.path.isdir(self.input_folder):
                    self.image_paths = [
                        os.path.join(self.input_folder, f)
                        for f in os.listdir(self.input_folder)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                    ]
                    if self.image_paths:
                        self.process_button.config(state=tk.NORMAL)

    def select_input_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.input_folder = folder_path
            self.image_paths = [
                os.path.join(self.input_folder, f)
                for f in os.listdir(self.input_folder)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            if not self.image_paths:
                messagebox.showerror("Error", "No valid images found in the selected folder.")
            else:
                messagebox.showinfo("Input Folder", f"Input folder set to: {folder_path}")
                self.process_button.config(state=tk.NORMAL)
                self.save_config()

    def select_output_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.output_folder = folder_path
            messagebox.showinfo("Output Folder", f"Output folder set to: {folder_path}")
            self.save_config()

    def display_image(self, image, canvas_type="before"):
        canvas_width = self.before_canvas.winfo_width() if canvas_type == "before" else self.after_canvas.winfo_width()
        canvas_height = self.before_canvas.winfo_height() if canvas_type == "before" else self.after_canvas.winfo_height()

        if isinstance(image, str):
            img = Image.open(image)
        else:
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        img = img.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)

        canvas = self.before_canvas if canvas_type == "before" else self.after_canvas

        canvas.delete("all")
        canvas.create_image(0, 0, image=img_tk, anchor="nw")
        canvas.image = img_tk

    def process_folder(self):
        if not self.input_folder or not self.output_folder:
            messagebox.showerror("Error", "Please select both input and output folders in the Settings tab.")
            return

        total_images = len(self.image_paths)
        for idx, image_path in enumerate(self.image_paths):
            self.display_image(image_path, "before")

            results = run_inference(
                model=self.model,
                source=image_path,
                conf_thres=0.5,
                iou_thres=0.45
            )

            annotated_images_batch = annotate_images_batch(results)

            for path, annotated_img in annotated_images_batch:
                detections_folder = "detections" if results[0][1] is not None else "no_detections"
                subfolder_path = os.path.join(self.output_folder, detections_folder)
                os.makedirs(subfolder_path, exist_ok=True)

                self.display_image(annotated_img, "after")

                output_path = os.path.join(subfolder_path, os.path.basename(path))
                cv2.imwrite(output_path, annotated_img)

            self.progress_bar["value"] = (idx + 1) / total_images * 100
            self.root.update_idletasks()
            self.show_gallery_images()

        messagebox.showinfo("Processing Complete", f"Processed images saved to {self.output_folder}.")

    def explorer_view(self, output_image_path):
        if os.name == 'nt':
            os.startfile(output_image_path)
    
    def on_resize(self, event):
        if event.widget == self.root:
            self.before_canvas.config(width=event.width // 2, height=event.height // 2)
            self.after_canvas.config(width=event.width // 2, height=event.height // 2)
            self.calculate_images_per_row()
            self.show_gallery_images()

if __name__ == "__main__":
    from yolov7.seg.segment.custom_predict import load_model

    model, _, _, _ = load_model(
        weights=r"yolov7\seg\runs\train-seg\custom2\weights\best.pt",
        device="cpu",
        data=r"yolov7\seg\crack-3\data.yaml"
    )
    model.eval()

    root = tk.Tk()
    app = YOLOv7App(root, model)
    root.mainloop()
