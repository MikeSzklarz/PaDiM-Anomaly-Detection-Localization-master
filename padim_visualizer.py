"""
A simple, fast PaDiM visualizer using Python's built-in tkinter library.
Two-pane view:
  Left  = Original + segmentation OUTLINE (percentile threshold)
  Right = Original + heatmap overlay

Dependencies:
  pip install pillow matplotlib
(Assumes you already have torch, torchvision, numpy, scipy from your project)
"""
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import matplotlib.cm as cm
import os
import glob
import pickle
import random
import io

import torch
import torch.nn.functional as F
from torchvision.models import wide_resnet50_2, resnet18, efficientnet_b5, EfficientNet_B5_Weights
from torchvision import transforms as T
import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion
from scipy.spatial.distance import mahalanobis

# --- Helper Functions (from your provided files) ---

# Global list to store intermediate features
INTERMEDIATE_FEATURE_MAPS = []

def hook_function(module, input, output):
    """A simple hook that appends the output of a layer to a global list."""
    INTERMEDIATE_FEATURE_MAPS.append(output)

def denormalize_image_for_display(tensor_image):
    """Reverses ImageNet normalization for display (from misc.py)"""
    x = tensor_image.cpu().clone().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    denormalized_img = (
        ((x.transpose(1, 2, 0) * std) + mean) * 255.0
    ).astype(np.uint8)
    return Image.fromarray(denormalized_img)

def concatenate_embeddings(larger_map, smaller_map):
    """Aligns and concatenates two feature maps (from embeddings.py)"""
    b, c1, h1, w1 = larger_map.size()
    _, c2, h2, w2 = smaller_map.size()
    stride = int(h1 / h2)
    unfolded = F.unfold(larger_map, kernel_size=stride, dilation=1, stride=stride)
    unfolded = unfolded.view(b, c1, -1, h2, w2)
    output_tensor = torch.zeros(
        b, c1 + c2, unfolded.size(2), h2, w2, device=larger_map.device
    )
    for i in range(unfolded.size(2)):
        patch = unfolded[:, :, i, :, :]
        output_tensor[:, :, i, :, :] = torch.cat((patch, smaller_map), 1)
    output_tensor = output_tensor.view(b, -1, h2 * w2)
    final_embedding = F.fold(
        output_tensor, kernel_size=stride, output_size=(h1, w1), stride=stride
    )
    return final_embedding

# --- Fast overlay utilities (no pyplot figures) ---

def heatmap_overlay_pil(orig_pil, heatmap_01, alpha=0.45, cmap_name="jet"):
    """
    Convert [0,1] heatmap to colored RGBA and alpha-blend on top of orig_pil.
    Returns a PIL.Image (RGB).
    """
    h, w = heatmap_01.shape
    # Colorize via matplotlib colormap -> RGBA uint8
    cmap = cm.get_cmap(cmap_name)
    colored = (cmap(heatmap_01) * 255).astype(np.uint8)  # (H,W,4)
    heat_rgba = Image.fromarray(colored, mode="RGBA")

    # Ensure same size as orig
    if heat_rgba.size != orig_pil.size:
        heat_rgba = heat_rgba.resize(orig_pil.size, Image.NEAREST)

    # Blend: scale heatmap alpha by 'alpha'
    # Multiply existing alpha channel by desired alpha
    r, g, b, a = heat_rgba.split()
    a = a.point(lambda px: int(px * alpha))
    heat_rgba = Image.merge("RGBA", (r, g, b, a))

    # Composite onto original
    base = orig_pil.convert("RGBA")
    out = Image.alpha_composite(base, heat_rgba).convert("RGB")
    return out

def segmentation_outline_overlay_pil(orig_pil, mask_bool, color=(0, 255, 0), thickness=2):
    """
    Draw a thin outline (contour) from a binary mask onto the original image.
    No skimage dependency; uses binary morphology to get edges.
    """
    # Resize mask to orig size if needed
    mh, mw = mask_bool.shape
    if (mw, mh) != orig_pil.size:
        mask_img = Image.fromarray((mask_bool * 255).astype(np.uint8), mode="L")
        mask_img = mask_img.resize(orig_pil.size, Image.NEAREST)
        mask_bool = np.array(mask_img) > 0

    # Edge = dilated XOR eroded (cheap, effective)
    dil = binary_dilation(mask_bool, iterations=thickness)
    ero = binary_erosion(mask_bool, iterations=thickness)
    edge = np.logical_and(dil, np.logical_not(ero))

    out = orig_pil.copy()
    arr = np.array(out)  # (H,W,3) uint8
    # Color edges
    r, g, b = color
    arr[edge] = [r, g, b]
    return Image.fromarray(arr, mode="RGB")

# --- Backend Inference Logic ---

def load_model_and_distribution(architecture, pkl_path, resize_val, crop_val):
    """Loads the selected model, registers hooks, and loads the distribution."""
    global INTERMEDIATE_FEATURE_MAPS
    INTERMEDIATE_FEATURE_MAPS = []  # Clear hooks list

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Model and set up hooks (from run_experiment.py)
    if architecture == "wide_resnet50_2":
        model = wide_resnet50_2(weights="DEFAULT")
        total_dim, reduced_dim = 1792, 550
        model.layer1[-1].register_forward_hook(hook_function)
        model.layer2[-1].register_forward_hook(hook_function)
        model.layer3[-1].register_forward_hook(hook_function)
    elif architecture == "resnet18":
        model = resnet18(weights="DEFAULT")
        total_dim, reduced_dim = 448, 100
        model.layer1[-1].register_forward_hook(hook_function)
        model.layer2[-1].register_forward_hook(hook_function)
        model.layer3[-1].register_forward_hook(hook_function)
    elif architecture == "efficientnet_b5":
        model = efficientnet_b5(weights=EfficientNet_B5_Weights.DEFAULT)
        total_dim, reduced_dim = 472, 200  # Adjust to match training
        model.features[2].register_forward_hook(hook_function)
        model.features[4].register_forward_hook(hook_function)
        model.features[6].register_forward_hook(hook_function)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    model.to(device)
    model.eval()

    # 2. Load Distribution
    with open(pkl_path, 'rb') as f:
        distribution = pickle.load(f)  # Should be [mean, cov] or similar

    # 3. Create Transforms
    data_transforms = T.Compose([
        T.Resize(resize_val, Image.Resampling.LANCZOS),
        T.CenterCrop(crop_val),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 4. Set up random feature indices (using seed for reproducibility)
    random.seed(1024)
    torch.manual_seed(1024)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1024)

    random_feature_indices = torch.tensor(
        random.sample(range(0, total_dim), reduced_dim)
    ).to(device)

    return model, distribution, data_transforms, random_feature_indices, device

def run_inference(model, distribution, transforms, random_indices, device, image_path):
    """Runs a single image through the PaDiM pipeline to get an anomaly map."""
    global INTERMEDIATE_FEATURE_MAPS
    INTERMEDIATE_FEATURE_MAPS = []  # Clear hooks

    img = Image.open(image_path).convert('RGB')
    x = transforms(img).unsqueeze(0).to(device)

    with torch.no_grad():
        _ = model(x)

    embedding_vectors = INTERMEDIATE_FEATURE_MAPS[0]
    for i in range(1, len(INTERMEDIATE_FEATURE_MAPS)):
        embedding_vectors = concatenate_embeddings(embedding_vectors, INTERMEDIATE_FEATURE_MAPS[i])

    embedding_vectors = torch.index_select(embedding_vectors, 1, random_indices)

    B, C, H, W = embedding_vectors.size()
    embedding_vectors = embedding_vectors.view(B, C, H * W).cpu().numpy()

    # Unpack distribution
    train_mean, train_cov = distribution[0], distribution[1]

    # Cache inverse covariance once per session (on the function)
    try:
        if not hasattr(run_inference, "train_cov_inv"):
            print("Calculating inverse covariance matrices... (one-time)")
            # train_cov shape expected: (C, C, H*W)
            train_cov_inv = np.linalg.inv(train_cov.transpose(2, 0, 1)).transpose(1, 2, 0)
            run_inference.train_cov_inv = train_cov_inv
        else:
            train_cov_inv = run_inference.train_cov_inv

    except np.linalg.LinAlgError:
        print("Calculating inverse covariance with identity fallback...")
        I = np.identity(C)
        train_cov_inv = np.zeros_like(train_cov)
        for i in range(H * W):
            train_cov_inv[:, :, i] = np.linalg.inv(train_cov[:, :, i] + 0.01 * I)
        run_inference.train_cov_inv = train_cov_inv

    # Mahalanobis per spatial position
    dist_list = []
    for i in range(H * W):
        mean = train_mean[:, i]
        conv_inv = train_cov_inv[:, :, i]
        dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
        dist_list.append(dist)

    dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)

    # Upsample to input size
    dist_list = torch.tensor(dist_list)
    score_map = F.interpolate(
        dist_list.unsqueeze(1),
        size=x.size(2),
        mode='bilinear',
        align_corners=False
    ).squeeze().numpy()

    # Smooth + [0,1] normalize
    score_map = gaussian_filter(score_map, sigma=4)
    max_score = score_map.max()
    min_score = score_map.min()
    if max_score == min_score:
        score_map = np.zeros_like(score_map)
    else:
        score_map = (score_map - min_score) / (max_score - min_score)

    return x.squeeze(0), score_map  # transformed tensor, normalized [0,1] map

# --- Tkinter GUI Application ---

class PaDiMVisualizer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PaDiM Visualizer")
        self.geometry("1200x640")

        # --- State Variables ---
        self.model = None
        self.distribution = None
        self.transforms = None
        self.indices = None
        self.device = None
        self.image_files = []
        self.current_index = 0

        # Cached per-image artifacts for speed
        self.current_score_map = None            # np.ndarray [H,W] in [0,1]
        self.current_original_tensor = None      # torch tensor (C,H,W)
        self.orig_pil_full = None                # PIL original (denormalized)
        self.orig_pil_resized = None             # PIL resized to display
        self.heat_overlay_resized = None         # PIL heat overlay (orig + heat)

        # Display size
        self.display_size = (480, 480)

        # --- Top Control Frame ---
        control_frame = ttk.Frame(self)
        control_frame.pack(pady=8)

        ttk.Label(control_frame, text="Model:").pack(side=tk.LEFT, padx=5)
        self.arch_var = tk.StringVar(value="wide_resnet50_2")
        ttk.OptionMenu(control_frame, self.arch_var, "wide_resnet50_2",
                       "wide_resnet50_2", "resnet18", "efficientnet_b5").pack(side=tk.LEFT)

        ttk.Label(control_frame, text="Crop:").pack(side=tk.LEFT, padx=5)
        self.crop_var = tk.StringVar(value="256")
        ttk.Entry(control_frame, textvariable=self.crop_var, width=5).pack(side=tk.LEFT)

        ttk.Button(control_frame, text="Load Model & Distribution", command=self.load_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Load Image Folder", command=self.load_images).pack(side=tk.LEFT, padx=5)

        # --- Image Display Frame (2 panes) ---
        image_frame = ttk.Frame(self)
        image_frame.pack(pady=8, fill=tk.BOTH, expand=True)
        image_frame.grid_columnconfigure((0, 1), weight=1)

        ttk.Label(image_frame, text="Segmentation Outline", font=("Arial", 13)).grid(row=0, column=0)
        self.left_label = ttk.Label(image_frame, borderwidth=2, relief="sunken")
        self.left_label.grid(row=1, column=0, padx=10, pady=10)

        ttk.Label(image_frame, text="Heatmap Overlay", font=("Arial", 13)).grid(row=0, column=1)
        self.right_label = ttk.Label(image_frame, borderwidth=2, relief="sunken")
        self.right_label.grid(row=1, column=1, padx=10, pady=10)

        # --- Bottom Navigation Frame ---
        nav_frame = ttk.Frame(self)
        nav_frame.pack(pady=8, fill=tk.X)
        nav_frame.grid_columnconfigure(1, weight=1)

        ttk.Button(nav_frame, text="< Previous", command=self.prev_image).grid(row=0, column=0, padx=10)

        self.slider_label = tk.StringVar(value="Percentile Threshold: 95.0")
        self.slider = ttk.Scale(nav_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self.update_segmentation_overlay)
        self.slider.set(95.0)
        self.slider.grid(row=0, column=1, padx=10, sticky="ew")
        ttk.Label(nav_frame, textvariable=self.slider_label).grid(row=1, column=1)

        ttk.Button(nav_frame, text="Next >", command=self.next_image).grid(row=0, column=2, padx=10)

        self.status_label = ttk.Label(self, text="Status: Load a model and images to begin.")
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X, ipady=5)

    def load_model(self):
        pkl_path = filedialog.askopenfilename(title="Select .pkl distribution file",
                                              filetypes=[("Pickle files", "*.pkl")])
        if not pkl_path:
            return

        try:
            arch = self.arch_var.get()
            crop = int(self.crop_var.get())
            resize = crop  # keep simple; change here if you want decoupled resize/crop

            self.status_label.config(text=f"Loading {arch} and distribution... Please wait.")
            self.update_idletasks()

            self.model, self.distribution, self.transforms, self.indices, self.device = \
                load_model_and_distribution(arch, pkl_path, resize, crop)

            self.status_label.config(text="Model loaded successfully. Ready to load images.")

        except Exception as e:
            messagebox.showerror("Error Loading Model", str(e))
            self.status_label.config(text="Error loading model.")

    def load_images(self):
        if not self.model:
            messagebox.showwarning("Model Not Loaded", "Please load a model and distribution first.")
            return

        folder_path = filedialog.askdirectory(title="Select Image Folder")
        if not folder_path:
            return

        img_files = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
            img_files.extend(glob.glob(os.path.join(folder_path, ext)))

        self.image_files = sorted(img_files)
        if not self.image_files:
            messagebox.showinfo("No Images Found", "No compatible images found in that folder.")
            return

        self.current_index = 0
        self.process_and_display_current_image()

    def process_and_display_current_image(self):
        if not self.image_files:
            return

        img_path = self.image_files[self.current_index]

        try:
            # Run the full inference
            self.current_original_tensor, self.current_score_map = \
                run_inference(self.model, self.distribution, self.transforms,
                              self.indices, self.device, img_path)

            # Cache original PILs (full + resized)
            self.orig_pil_full = denormalize_image_for_display(self.current_original_tensor)  # full crop size
            self.orig_pil_resized = self.orig_pil_full.resize(self.display_size, Image.NEAREST)

            # Build RIGHT heatmap overlay once per image
            heat_right = heatmap_overlay_pil(self.orig_pil_resized, 
                                             self.current_score_map, 
                                             alpha=0.45, 
                                             cmap_name="jet")
            self.heat_overlay_resized = heat_right  # cache

            # Push right pane
            self.right_photo = ImageTk.PhotoImage(self.heat_overlay_resized)
            self.right_label.config(image=self.right_photo)
            self.right_label.image = self.right_photo

            # LEFT segmentation outline (depends on slider)
            self.update_segmentation_overlay()

            self.status_label.config(
                text=f"Displaying: {os.path.basename(img_path)} "
                     f"({self.current_index + 1}/{len(self.image_files)})"
            )

        except Exception as e:
            messagebox.showerror("Inference Error", f"Failed to process {img_path}:\n{e}")
            self.status_label.config(text=f"Error processing {os.path.basename(img_path)}.")

    def update_segmentation_overlay(self, *args):
        """Only updates the LEFT outline overlay based on the percentile slider."""
        if self.current_score_map is None or self.orig_pil_resized is None:
            return

        percentile = self.slider.get()
        self.slider_label.set(f"Percentile Threshold: {percentile:.1f}")

        try:
            threshold_value = np.percentile(self.current_score_map, percentile)
            mask = (self.current_score_map > threshold_value)

            # Outline on top of original resized
            left_img = segmentation_outline_overlay_pil(
                self.orig_pil_resized, mask, color=(0, 255, 0), thickness=2
            )
            self.left_photo = ImageTk.PhotoImage(left_img)
            self.left_label.config(image=self.left_photo)
            self.left_label.image = self.left_photo

        except Exception as e:
            print(f"Error updating segmentation overlay: {e}")  # non-critical

    def prev_image(self):
        if not self.image_files:
            return
        self.current_index = (self.current_index - 1) % len(self.image_files)
        self.process_and_display_current_image()

    def next_image(self):
        if not self.image_files:
            return
        self.current_index = (self.current_index + 1) % len(self.image_files)
        self.process_and_display_current_image()

if __name__ == "__main__":
    # Set up WSL2 GUI display if needed
    if "WSL_DISTRO_NAME" in os.environ:
        if "DISPLAY" not in os.environ:
            os.environ["DISPLAY"] = "localhost:0.0"
            print("Set DISPLAY='localhost:0.0' for WSL2. Make sure your X-server is running.")

    app = PaDiMVisualizer()
    app.mainloop()
