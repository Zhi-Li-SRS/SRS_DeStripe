import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy import ndimage
from scipy.signal import find_peaks, medfilt, savgol_filter


def vistamap_tissue_isolation(image):
    """
    Isolate tissue using texture filtering with a disk structuring element
    """
    disk_size = max(50, int(0.01 * min(image.shape)))
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (disk_size, disk_size))

    closed_img = cv2.morphologyEx(image, cv2.MORPH_CLOSE, disk)
    small_disk = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (max(1, int(0.001 * image.shape[0])), max(1, int(0.001 * image.shape[0])))
    )
    texture_img = cv2.morphologyEx(closed_img, cv2.MORPH_OPEN, small_disk)

    threshold = 0.1 * np.max(texture_img)
    binary_map = (texture_img > threshold).astype(np.uint8)

    se = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (max(1, int(0.0001 * image.shape[0])), max(1, int(0.0001 * image.shape[0])))
    )
    binary_map = cv2.morphologyEx(binary_map, cv2.MORPH_CLOSE, se)
    binary_map = ndimage.binary_fill_holes(binary_map).astype(np.uint8)

    return binary_map


def handle_extreme_pixels(image, mask):
    """
    Handle extreme pixel values:
    - Replace NaN and Inf values
    - Temporarily adjust top 1% of pixel intensities
    """
    processed_img = image.copy()

    processed_img = np.nan_to_num(
        processed_img, nan=0, posinf=np.max(processed_img[np.isfinite(processed_img)])
    )

    valid_pixels = processed_img[mask > 0]
    if len(valid_pixels) > 0:
        p99_threshold = np.percentile(valid_pixels, 99)

        bright_mask = processed_img > p99_threshold

        mean_tissue = np.mean(
            processed_img[(processed_img > 0) & (processed_img <= p99_threshold) & (mask > 0)]
        )
        temp_img = processed_img.copy()
        temp_img[bright_mask] = mean_tissue

        return temp_img, bright_mask, processed_img

    return processed_img, np.zeros_like(processed_img, dtype=bool), processed_img


def vistamap_envelope(profile, window_size):
    """
    Calculate envelope for intensity profile using VISTAmap approach
    """
    # Apply median filtering
    profile = medfilt(profile, kernel_size=min(11, len(profile) - (1 if len(profile) % 2 == 0 else 0)))
    # Calculate upper and lower envelopes
    upper_env = ndimage.maximum_filter1d(profile, size=window_size)
    lower_env = ndimage.minimum_filter1d(profile, size=window_size)

    # Smooth envelopes if window size is large enough
    if window_size >= 5:
        sg_window = min(window_size, len(profile) - 1)
        sg_window = sg_window if sg_window % 2 == 1 else sg_window - 1
        if sg_window >= 5:
            upper_env = savgol_filter(upper_env, sg_window, 3)
            lower_env = savgol_filter(lower_env, sg_window, 3)

    ideal_line = (upper_env + lower_env) / 2

    return lower_env, upper_env, ideal_line


def calculate_vistamap_multiplier(profile, ideal_line):
    """
    Calculate correction multiplier
    """
    multiplier = np.ones_like(profile)
    non_zero = profile > 1e-6

    # Calculate the correction multiplier
    multiplier[non_zero] = ideal_line[non_zero] / profile[non_zero]
    multiplier = np.nan_to_num(multiplier, nan=1.0, posinf=2.5, neginf=1.0)

    peaks, _ = find_peaks(multiplier)
    if len(peaks) > 0:
        mean_peak_height = np.mean(multiplier[peaks])

        # Constrain multiplier values
        multiplier[multiplier < 1.0] = 1.0  # Do not process bright region
        multiplier[multiplier > mean_peak_height] = mean_peak_height  # Limit to mean peak height

    return multiplier


def apply_final_morphological_operations(image, mask):
    """
    Apply final morphological operations with line structuring elements
    """
    processed = image.copy()

    disk50 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
    tophat = cv2.morphologyEx(processed, cv2.MORPH_TOPHAT, disk50)  # Top-hat filtering
    bright_spots = tophat > (tophat.mean() + 3 * tophat.std())  # Detect bright spots
    image_without_points = processed.copy()
    image_without_points[bright_spots] = processed.mean()  # Replace bright spots with mean

    # horizontal line operations
    h_line = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    closed = cv2.morphologyEx(image_without_points, cv2.MORPH_CLOSE, h_line)  # Close horizontal lines
    illumination_h = cv2.morphologyEx(closed, cv2.MORPH_OPEN, h_line)

    # vertical line operations after horizontalline
    v_line = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
    closed = cv2.morphologyEx(illumination_h, cv2.MORPH_CLOSE, v_line)
    illumination = cv2.morphologyEx(closed, cv2.MORPH_OPEN, v_line)

    illumination = np.maximum(illumination, 1e-6)
    result = np.zeros_like(processed)
    non_zero = illumination > 1e-6
    result[non_zero] = processed[non_zero] / illumination[non_zero] * np.mean(illumination)

    result = np.clip(result, 0, 1)

    result = result * mask

    return result


class VISTAmapDestriper:
    """
    Implementation of VISTAmap (VIgnetted Stitched-Tile Adjustment using Morphological Adaptive Processing)
    """

    def __init__(self, tile_size=256):
        self.tile_size = tile_size

    def load_and_preprocess(self, image_path, mask_path=None):
        """
        Load and preprocess the image and mask.
        If mask_path is not provided, will automatically generate a mask.
        """
        original_image = tifffile.imread(image_path)

        assert original_image.ndim in [2, 3], "Image must be 2D or 3D (single channel or multi-channel)."
        # If 3D, convert to 2D by squeezing
        if original_image.ndim > 2:
            original_image = original_image.squeeze()

        non_zero_pixels = original_image[original_image > 0]
        if len(non_zero_pixels) > 0:
            image_min, image_max = np.percentile(non_zero_pixels, (2, 98))
        else:
            image_min, image_max = np.min(original_image), np.max(original_image)

        normalized_image = np.clip((original_image - image_min) / (image_max - image_min), 0, 1)

        # Generate mask if not provided
        if mask_path is None or not os.path.exists(mask_path):
            print("Mask not provided. Generating mask using texture-based isolation.")
            mask = vistamap_tissue_isolation(normalized_image)
        else:
            mask = tifffile.imread(mask_path)
            if mask.ndim > 2:
                mask = mask.squeeze()
            mask = (mask > 0).astype(np.uint8)

        temp_image, bright_mask, orig_normalized = handle_extreme_pixels(normalized_image, mask)

        return temp_image, mask, bright_mask, orig_normalized, (image_min, image_max)

    def detect_stripes_fft(self, image, mask, orientation="vertical"):
        """
        Detect stripes using FFT analysis.
        If tile_size is provided, use it instead of FFT detection.
        """
        img_length = image.shape[1] if orientation == "vertical" else image.shape[0]

        if self.tile_size is not None:
            print(f"Using specified tile size: {self.tile_size}")
            tile_width = self.tile_size
            dominant_freq = 1 / tile_width

            # Calculate number of tiles and edge positions
            num_tiles = max(1, int(img_length / tile_width))
            edge_positions = []

            # Create edge positions at tile boundaries
            for i in range(num_tiles + 1):
                pos = i * tile_width
                if pos <= img_length:
                    edge_positions.append(pos)
                else:
                    edge_positions.append(img_length)

            edge_positions = np.array(edge_positions)

            return edge_positions, dominant_freq

        if orientation == "vertical":
            profile = np.nanmean(image * mask, axis=0)
            profile[np.isnan(profile)] = np.nanmean(profile)
        else:
            profile = np.nanmean(image * mask, axis=1)
            profile[np.isnan(profile)] = np.nanmean(profile)

        profile = medfilt(profile, kernel_size=11)

        # Compute FFT
        fft_result = np.fft.fft(profile)
        n = len(profile)
        frequencies = np.fft.fftfreq(n)

        # Analyze only positive frequencies
        half_n = n // 2
        frequencies = frequencies[:half_n]
        magnitude = np.abs(fft_result[:half_n]) / n

        start_idx = max(1, int(0.001 * half_n))
        peaks, _ = find_peaks(magnitude[start_idx:], distance=5)
        peaks = peaks + start_idx

        if len(peaks) == 0:
            print(f"No peaks found in {orientation} FFT. Using evenly spaced divisions.")
            edge_positions = np.linspace(0, img_length - 1, 6, dtype=int)
            return edge_positions, None

        # Find dominant frequency
        dominant_peak = peaks[np.argmax(magnitude[peaks])]
        dominant_freq = frequencies[dominant_peak]

        # Calculate estimated stripe width
        stripe_width = int(abs(1 / dominant_freq))

        # Check on stripe width
        if stripe_width < 10 or stripe_width > min(image.shape) / 2:
            print(f"Warning: Unreasonable stripe width ({stripe_width}). Using evenly spaced divisions.")
            edge_positions = np.linspace(0, img_length - 1, 6, dtype=int)
            return edge_positions, 1 / (img_length / 5)  # Approx frequency

        # Calculate tile edge positions based on detected frequency
        num_tiles = max(5, int(img_length / stripe_width))
        edge_positions = np.linspace(0, img_length - 1, num_tiles + 1, dtype=int)

        return edge_positions, dominant_freq

    def remove_stripes_vistamap(self, image_path, mask_path=None, output_path="process.tif"):
        """Main method implementing the VISTAmap approach"""
        self.output_path = output_path

        # Create output directory
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Preprocess image
        temp_image, mask, bright_mask, orig_normalized, (image_min, image_max) = self.load_and_preprocess(
            image_path, mask_path
        )
        _, h_dominant_freq = self.detect_stripes_fft(temp_image, mask, orientation="horizontal")
        _, v_dominant_freq = self.detect_stripes_fft(temp_image, mask, orientation="vertical")

        # Process horizontal direction
        h_profile = np.nanmean(temp_image * mask, axis=0)
        h_profile[np.isnan(h_profile)] = np.nanmean(h_profile)

        if self.tile_size is not None:
            h_window_size = max(11, int(0.7 * self.tile_size))
        elif h_dominant_freq is not None:
            h_window_size = max(11, int(0.7 * (1 / abs(h_dominant_freq))))
        else:
            h_window_size = max(11, int(temp_image.shape[1] * 0.05))

        h_window_size = h_window_size + 1 if h_window_size % 2 == 0 else h_window_size
        h_lower, h_upper, h_ideal = vistamap_envelope(h_profile, h_window_size)
        h_multiplier = calculate_vistamap_multiplier(h_profile, h_ideal)
        h_corrected = temp_image * h_multiplier[np.newaxis, :]

        # Process vertical direction
        v_profile = np.nanmean(h_corrected * mask, axis=1)
        v_profile[np.isnan(v_profile)] = np.nanmean(v_profile)

        if self.tile_size is not None:
            v_window_size = max(11, int(0.7 * self.tile_size))
        elif v_dominant_freq is not None:
            v_window_size = max(11, int(0.7 * (1 / abs(v_dominant_freq))))
        else:
            v_window_size = max(11, int(temp_image.shape[0] * 0.05))

        v_window_size = v_window_size + 1 if v_window_size % 2 == 0 else v_window_size
        v_lower, v_upper, v_ideal = vistamap_envelope(v_profile, v_window_size)
        v_multiplier = calculate_vistamap_multiplier(v_profile, v_ideal)
        v_corrected = h_corrected * v_multiplier[:, np.newaxis]

        # Apply final morphological operations
        final_image = apply_final_morphological_operations(v_corrected, mask)

        # Restore original bright pixels
        final_image[bright_mask] = orig_normalized[bright_mask]

        # Rescale to original intensity range
        final_image = final_image * (image_max - image_min) + image_min

        # Compare original and final images
        original = tifffile.imread(image_path)
        if original.ndim > 2:
            original = original.squeeze()

        plt.figure(figsize=(15, 10))
        plt.subplot(121)
        plt.imshow(original, cmap="gray")
        plt.title("Original Image")

        plt.subplot(122)
        plt.imshow(final_image, cmap="gray")
        plt.title("Corrected Image")
        comparison_path = os.path.splitext(output_path)[0] + "_compare.png"
        plt.savefig(comparison_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Save the final image
        tifffile.imwrite(output_path, final_image.astype(np.float32))
        print(f"Processed image saved to: {output_path}")

        return final_image


def main():
    """Main function to run the script from command line"""
    parser = argparse.ArgumentParser(
        description="VISTAmap: VIgnetted Stitched-Tile Adjustment using Morphological Adaptive Processing"
    )
    parser.add_argument("--image_path", type=str, default="791.tif", help="Path to the image file.")
    parser.add_argument(
        "--mask_path",
        type=str,
        default="srs_mask.tif",
        help="Better to have the path to the mask file. If not provided, a mask will be generated automatically.",
    )
    parser.add_argument(
        "--output_path", type=str, default="output/process.tif", help="Path to save the output image."
    )
    parser.add_argument(
        "--tile_size", type=int, default=None, help="Optional to specify the size of each tile (e.g., 256)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: Input image file not found: {args.image_path}")
        return

    if args.mask_path is not None and not os.path.exists(args.mask_path):
        print(f"Warning: Mask file not found: {args.mask_path}. Will generate automatically.")
        args.mask_path = None

    print(f"Starting VISTAmap processing on {args.image_path}")
    processor = VISTAmapDestriper(tile_size=args.tile_size)

    processor.remove_stripes_vistamap(args.image_path, args.mask_path, args.output_path)

    print("Processing completed successfully!")


if __name__ == "__main__":
    main()
