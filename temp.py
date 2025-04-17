import cv2
import numpy as np
from PIL import Image

# Image paths (you can reorder these if needed)
image_paths = [
    "./image2/reward_vs_time_step_all_0.01_errorbar.png",  
    "./image2/reward_vs_time_step_all_0.02_errorbar.png",  
    "./image2/reward_vs_time_step_all_0.03_errorbar.png",  
    "./image2/reward_vs_time_step_all_0.04_errorbar.png"   
]

# Correct colors: Purple, Green, Blue, Orange (in reverse)
tint_colors = [
    [224, 163, 46],   
    [70, 148, 73],      
    [214, 126, 44],      
    [187, 86, 149]     
]

# Load base image
bgr_base = cv2.imread(image_paths[0])
rgb_base = cv2.cvtColor(bgr_base, cv2.COLOR_BGR2RGB)
height, width = rgb_base.shape[:2]
rgba_base = np.dstack((rgb_base, 255 * np.ones((height, width), dtype=np.uint8)))
img_base = Image.fromarray(rgba_base, "RGBA")

# HSV range for detecting the base plot line
lower_line = np.array([100, 50, 50])
upper_line = np.array([140, 255, 255])

# Overlay tinted lines
for i in range(1, len(image_paths)):
    bgr = cv2.imread(image_paths[i])
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (width, height))

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    line_mask = cv2.inRange(hsv, lower_line, upper_line)

    tinted = rgb.copy()
    tinted[line_mask > 0] = tint_colors[i]

    alpha = np.zeros_like(line_mask)
    alpha[line_mask > 0] = 255  # Full opacity

    rgba = np.dstack((tinted, alpha)).astype(np.uint8)
    img_overlay = Image.fromarray(rgba, "RGBA")

    img_base = Image.alpha_composite(img_base, img_overlay)

# Save final output
img_base.save("final_overlay_learning_rate_colored.png")
print("✅ Saved: 'final_overlay_learning_rate_colored.png'")
