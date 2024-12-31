import os
import random

import cv2
import numpy as np
from PIL import Image, ImageEnhance


def augment_image(image):
    augmented_images = []

    # Ensure the image is in RGBA mode to preserve transparency
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    # Rotate
    for angle in [90, 180, 270]:
        rotated = image.rotate(angle, expand=True)
        augmented_images.append(rotated)

    # Flip
    augmented_images.append(image.transpose(Image.FLIP_LEFT_RIGHT))
    augmented_images.append(image.transpose(Image.FLIP_TOP_BOTTOM))

    # Brightness adjustment
    rgb_image = image.convert("RGB")
    enhancer = ImageEnhance.Brightness(rgb_image)
    brightened = enhancer.enhance(random.uniform(0.5, 1.5))
    brightened = Image.alpha_composite(image, brightened.convert("RGBA"))
    augmented_images.append(brightened)

    # Add noise
    np_image = np.array(image)
    noise = np.random.randint(0, 50, np_image[:, :, :3].shape, dtype="uint8")
    noisy_image = np_image.copy()
    noisy_image[:, :, :3] = cv2.add(np_image[:, :, :3], noise)
    augmented_images.append(Image.fromarray(noisy_image, "RGBA"))

    # Scaling
    scale_factor = random.uniform(0.8, 1.2)
    w, h = image.size
    scaled = image.resize((int(w * scale_factor), int(h * scale_factor)))
    scaled = scaled.resize((w, h), Image.Resampling.LANCZOS)
    augmented_images.append(scaled)

    # Distortions
    augmented_images += apply_distortions(image)

    # New Augmentations
    augmented_images += apply_additional_augmentations(image)

    return augmented_images


def apply_distortions(image):
    """Apply distortions to the image."""
    distorted_images = []
    np_image = np.array(image)

    # Random perspective transformation
    h, w = np_image.shape[:2]
    for _ in range(3):  # Add 3 distorted versions
        src_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        dst_pts = src_pts + np.random.uniform(-30, 30, src_pts.shape).astype(np.float32)
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(
            np_image,
            matrix,
            (w, h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        distorted_images.append(Image.fromarray(warped, "RGBA"))

    # Elastic distortions
    for _ in range(3):  # Add 3 elastic distortions
        alpha = random.randint(10, 30)  # Intensity of distortion
        sigma = random.randint(4, 6)  # Smoothing of distortion
        dx = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1), (15, 15), sigma) * alpha
        dy = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1), (15, 15), sigma) * alpha
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)
        elastically_distorted = cv2.remap(
            np_image,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        distorted_images.append(Image.fromarray(elastically_distorted, "RGBA"))

    return distorted_images


def apply_additional_augmentations(image):
    """Apply additional augmentations like blur, color jitter, cropping."""
    additional_images = []
    np_image = np.array(image)

    # Gaussian blur
    for _ in range(3):
        blurred = cv2.GaussianBlur(np_image, (5, 5), sigmaX=random.uniform(1.0, 2.0))
        additional_images.append(Image.fromarray(blurred, "RGBA"))

    # Color jitter (hue and saturation)
    for _ in range(3):
        hsv_image = cv2.cvtColor(np_image[:, :, :3], cv2.COLOR_RGB2HSV)
        hsv_image = np.array(hsv_image, dtype="float32")
        hsv_image[:, :, 0] += random.uniform(-10, 10)  # Hue
        hsv_image[:, :, 1] *= random.uniform(0.8, 1.2)  # Saturation
        hsv_image = np.clip(hsv_image, 0, 255).astype("uint8")
        jittered = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

        # Add alpha channel back to the jittered image
        alpha_channel = (
            np_image[:, :, 3]
            if np_image.shape[2] == 4
            else np.full((np_image.shape[0], np_image.shape[1]), 255, dtype="uint8")
        )
        jittered_rgba = np.dstack((jittered, alpha_channel))
        additional_images.append(Image.fromarray(jittered_rgba, "RGBA"))

    # Random cropping and resizing
    w, h = image.size
    for _ in range(3):
        crop_x = random.randint(0, w // 4)
        crop_y = random.randint(0, h // 4)
        cropped = image.crop((crop_x, crop_y, w - crop_x, h - crop_y))
        cropped_resized = cropped.resize((w, h), Image.Resampling.LANCZOS)
        additional_images.append(cropped_resized)

    return additional_images


# Function to process images in a folder
def process_folder(input_folder, output_folder):
    for subdir, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(".png"):
                input_path = os.path.join(subdir, file)
                output_subdir = os.path.join(
                    output_folder, os.path.relpath(subdir, input_folder)
                )
                os.makedirs(output_subdir, exist_ok=True)

                image = Image.open(input_path)
                augmented_images = augment_image(image)

                # Save original and augmented images
                base_name = os.path.splitext(file)[0]
                image.save(os.path.join(output_subdir, f"{base_name}_original.png"))
                for idx, aug_image in enumerate(augmented_images):
                    aug_image.save(
                        os.path.join(output_subdir, f"{base_name}_aug_{idx + 1}.png")
                    )


# Main function
if __name__ == "__main__":
    input_folder = "/Users/toby/Dev/what-do-we-think-of-tottenham/data/top-5-football-leagues"  # Replace with your input folder
    output_folder = "/Users/toby/Dev/what-do-we-think-of-tottenham/data/processed"  # Replace with your output folder
    os.makedirs(output_folder, exist_ok=True)

    process_folder(input_folder, output_folder)
