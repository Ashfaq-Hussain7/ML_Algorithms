#!pip install opencv-python numpy joblib

import cv2
import numpy as np
import time
from joblib import Parallel, delayed
import matplotlib.pyplot as plt


# Load the image
image_path = "input.jpg"  # Change this to your image path
image = cv2.imread(image_path)

# Check if the image is loaded correctly
if image is None:
    print("Error: Could not open or find the image.")
else:
    # Convert image to RGB for correct display in matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.title("Original Image")
    plt.axis("off")
    plt.show()


# Define convolution kernels for different operations
blur_kernel = np.array([[1/9, 1/9, 1/9],
                        [1/9, 1/9, 1/9],
                        [1/9, 1/9, 1/9]], dtype=np.float32)

sharpen_kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]], dtype=np.float32)

edge_kernel = np.array([[-1, -1, -1],
                        [-1,  8, -1],
                        [-1, -1, -1]], dtype=np.float32)


def apply_filter(image, kernel):
    """Applies a convolution filter to the image."""
    return cv2.filter2D(image, -1, kernel)


start_time = time.time()

# Apply filters sequentially
blurred_seq = apply_filter(image, blur_kernel)
sharpened_seq = apply_filter(image, sharpen_kernel)
edges_seq = apply_filter(image, edge_kernel)

end_time = time.time()
print(f"Sequential Execution Time: {end_time - start_time:.4f} sec")


start_time = time.time()

# Apply filters in parallel
blurred, sharpened, edges = Parallel(n_jobs=-1)(
    delayed(apply_filter)(image, kernel) for kernel in [blur_kernel, sharpen_kernel, edge_kernel]
)

end_time = time.time()
print(f"Parallel Execution Time: {end_time - start_time:.4f} sec")


# Convert images to RGB for correct display in Matplotlib
blurred_rgb = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
sharpened_rgb = cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)
edges_rgb = cv2.cvtColor(edges, cv2.COLOR_BGR2RGB)

# Plot images
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes[0].imshow(image_rgb)
axes[0].set_title("Original")
axes[1].imshow(blurred_rgb)
axes[1].set_title("Blurred")
axes[2].imshow(sharpened_rgb)
axes[2].set_title("Sharpened")
axes[3].imshow(edges_rgb, cmap="gray")
axes[3].set_title("Edge Detection")

for ax in axes:
    ax.axis("off")

plt.show()


cv2.imwrite("blurred.jpg", blurred)
cv2.imwrite("sharpened.jpg", sharpened)
cv2.imwrite("edge_detected.jpg", edges)
print("Processed images saved successfully!")

