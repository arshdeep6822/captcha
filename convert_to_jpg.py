import os
import cv2
IMAGE_FOLDER = "input/captcha_images_v2"

# Create a new folder for JPG images
OUTPUT_FOLDER = "input/jpg_images"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
for filename in os.listdir(IMAGE_FOLDER):
    if filename.endswith(".png"):
        # Read image
        img_path = os.path.join(IMAGE_FOLDER, filename)
        img = cv2.imread(img_path)

        # Convert filename to JPG
        new_filename = filename.replace(".png", ".jpg")
        new_path = os.path.join(OUTPUT_FOLDER, new_filename)

        # Save as JPG
        cv2.imwrite(new_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

print("✅ Conversion completed. All images are now in JPG format.")