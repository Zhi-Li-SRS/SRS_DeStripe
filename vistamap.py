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
    as described in the VISTAmap paper.
    """
    disk_size = max(50, int(0.05 * min(image.shape)))
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
    Handle extreme pixel values as described in VISTAmap:
    - Replace NaN and Inf values
    - Temporarily adjust top 1% of pixel intensities
    """
    processed_img = image.copy()

    # Handle NaN and Inf values
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
    # Apply median filtering to smooth the profile
    profile = medfilt(profile, kernel_size=min(11, len(profile) - (1 if len(profile) % 2 == 0 else 0)))

    # Calculate upper and lower envelopes
    upper_env = ndimage.maximum_filter1d(profile, size=window_size)
    lower_env = ndimage.minimum_filter1d(profile, size=window_size)

    # Smooth the envelopes
    if window_size >= 5:
        sg_window = min(window_size, len(profile) - 1)
        sg_window = sg_window if sg_window % 2 == 1 else sg_window - 1
        if sg_window >= 5:
            upper_env = savgol_filter(upper_env, sg_window, 3)
            lower_env = savgol_filter(lower_env, sg_window, 3)

    # Calculate ideal intensity (mean of upper and lower)
    ideal_line = (upper_env + lower_env) / 2

    return lower_env, upper_env, ideal_line


def calculate_vistamap_multiplier(profile, ideal_line):
    """
    Calculate correction multiplier with VISTAmap constraints
    """
    multiplier = np.ones_like(profile)
    non_zero = profile > 1e-6
    multiplier[non_zero] = ideal_line[non_zero] / profile[non_zero]

    multiplier = np.nan_to_num(multiplier, nan=1.0, posinf=2.0, neginf=1.0)

    peaks, _ = find_peaks(multiplier)
    if len(peaks) > 0:
        mean_peak_height = np.mean(multiplier[peaks])

        # Constrain multiplier values
        multiplier[multiplier < 1.0] = 1.0  # Keep peaks as they are
        multiplier[multiplier > mean_peak_height] = mean_peak_height  # Limit to mean peak height

    return multiplier


def apply_final_morphological_operations(image, mask):
    """
    Apply VISTAmap final morphological operations with line structuring elements
    """
    processed = image.copy()

    disk50 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
    tophat = cv2.morphologyEx(processed, cv2.MORPH_TOPHAT, disk50)
    bright_spots = tophat > (tophat.mean() + 3 * tophat.std())
    image_without_points = processed.copy()
    image_without_points[bright_spots] = processed.mean()

    # horizontal line operations
    h_line = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    closed = cv2.morphologyEx(image_without_points, cv2.MORPH_CLOSE, h_line)
    illumination_h = cv2.morphologyEx(closed, cv2.MORPH_OPEN, h_line)

    # vertical line operations
    v_line = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
    closed = cv2.morphologyEx(illumination_h, cv2.MORPH_CLOSE, v_line)
    illumination = cv2.morphologyEx(closed, cv2.MORPH_OPEN, v_line)

    # Ensure no division by zero
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
    for correcting vignetting artifacts in stitched microscopy images.
    """

    def __init__(self, debug=False, tile_size=256):
        self.debug = debug
        self.tile_size = tile_size

    def load_and_preprocess(self, image_path, mask_path=None):
        """
        Load and preprocess the image and mask.
        If mask_path is not provided, will automatically generate a mask.
        """
        original_image = tifffile.imread(image_path)

        if original_image.ndim > 2:
            original_image = original_image.squeeze()

        # Calculate percentiles for normalization
        non_zero_pixels = original_image[original_image > 0]
        if len(non_zero_pixels) > 0:
            image_min, image_max = np.percentile(non_zero_pixels, (2, 98))
        else:
            image_min, image_max = np.min(original_image), np.max(original_image)

        # Normalize image
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

        # Handle extreme pixel values
        temp_image, bright_mask, orig_normalized = handle_extreme_pixels(normalized_image, mask)

        return temp_image, mask, bright_mask, orig_normalized, (image_min, image_max)

    def detect_stripes_fft(self, image, mask, orientation="vertical"):
        """
        Detect stripes using FFT analysis based on VISTAmap approach.
        If tile_size is provided, use it instead of FFT detection.
        """
        img_length = image.shape[1] if orientation == "vertical" else image.shape[0]

        if self.tile_size is not None:
            print(f"Using user-specified tile size: {self.tile_size}")
            stripe_width = self.tile_size
            dominant_freq = 1 / stripe_width

            # Calculate number of tiles and edge positions
            num_tiles = max(1, int(img_length / stripe_width))
            edge_positions = []

            # Create edge positions at tile boundaries
            for i in range(num_tiles + 1):
                pos = i * stripe_width
                if pos <= img_length:
                    edge_positions.append(pos)
                else:
                    edge_positions.append(img_length)

            edge_positions = np.array(edge_positions)

            if self.debug:
                # Plot the intensity profile for debugging
                if orientation == "vertical":
                    profile = np.nanmean(image * mask, axis=0)
                else:
                    profile = np.nanmean(image * mask, axis=1)
                profile[np.isnan(profile)] = np.nanmean(profile)
                profile = medfilt(profile, kernel_size=11)

                plt.figure(figsize=(12, 6))
                plt.subplot(211)
                plt.plot(profile)
                for pos in edge_positions:
                    plt.axvline(x=pos, color="r", linestyle="--")
                plt.title(f"{orientation.capitalize()} Intensity Profile with Specified Tile Boundaries")

                plt.subplot(212)
                # Plot empty FFT since we're using specified tile size
                plt.text(
                    0.5,
                    0.5,
                    f"Using specified tile size: {stripe_width}",
                    horizontalalignment="center",
                    verticalalignment="center",
                )
                plt.title("FFT Analysis Not Used (Manual Tile Size)")
                plt.tight_layout()

                # Save the analysis plot
                output_dir = os.path.dirname(self.output_path)
                fft_plot_path = os.path.join(output_dir, f"{orientation}_manual_tile_analysis.png")
                plt.savefig(fft_plot_path)
                plt.close()

            return edge_positions, dominant_freq

        if orientation == "vertical":
            profile = np.nanmean(image * mask, axis=0)
            profile[np.isnan(profile)] = np.nanmean(profile)
        else:
            profile = np.nanmean(image * mask, axis=1)
            profile[np.isnan(profile)] = np.nanmean(profile)

        # Apply median filter
        profile = medfilt(profile, kernel_size=11)

        # Compute FFT
        fft_result = np.fft.fft(profile)
        n = len(profile)
        frequencies = np.fft.fftfreq(n)

        # Analyze only positive frequencies
        half_n = n // 2
        frequencies = frequencies[:half_n]
        magnitude = np.abs(fft_result[:half_n]) / n

        # Skip the DC component
        start_idx = max(1, int(0.001 * half_n))
        peaks, _ = find_peaks(magnitude[start_idx:], distance=5)
        peaks = peaks + start_idx

        if len(peaks) == 0:
            print(f"No peaks found in {orientation} FFT. Using evenly spaced divisions.")
            # Use evenly spaced divisions if no peaks found
            edge_positions = np.linspace(0, img_length - 1, 6, dtype=int)
            return edge_positions, None

        # Find dominant frequency
        dominant_peak = peaks[np.argmax(magnitude[peaks])]
        dominant_freq = frequencies[dominant_peak]

        # Calculate estimated stripe width
        stripe_width = int(abs(1 / dominant_freq))

        # Sanity check on stripe width
        if stripe_width < 10 or stripe_width > min(image.shape) / 2:
            print(f"Warning: Unreasonable stripe width ({stripe_width}). Using evenly spaced divisions.")
            edge_positions = np.linspace(0, img_length - 1, 6, dtype=int)
            return edge_positions, 1 / (img_length / 5)  # Approx frequency

        if self.debug:
            plt.figure(figsize=(12, 6))
            plt.subplot(211)
            plt.plot(profile)
            plt.title(f"{orientation.capitalize()} Intensity Profile")

            plt.subplot(212)
            plt.plot(frequencies, magnitude)
            plt.axvline(dominant_freq, color="r", linestyle="--")
            plt.title(f"FFT Magnitude Spectrum (Dominant period: {stripe_width} pixels)")
            plt.tight_layout()

            # Save the FFT analysis plot
            output_dir = os.path.dirname(self.output_path)
            fft_plot_path = os.path.join(output_dir, f"{orientation}_fft_analysis.png")
            plt.savefig(fft_plot_path)
            plt.close()

        # Calculate tile edge positions based on detected frequency
        num_tiles = max(5, int(img_length / stripe_width))
        edge_positions = np.linspace(0, img_length - 1, num_tiles + 1, dtype=int)

        return edge_positions, dominant_freq

    def detect_stripes_fft(self, image, mask, orientation="vertical"):
        """
        Detect stripes using FFT analysis based on VISTAmap approach
        """
        # Extract intensity profile of the masked region
        if orientation == "vertical":
            profile = np.nanmean(image * mask, axis=0)
            profile[np.isnan(profile)] = np.nanmean(profile)
        else:
            profile = np.nanmean(image * mask, axis=1)
            profile[np.isnan(profile)] = np.nanmean(profile)

        # Apply median filter
        profile = medfilt(profile, kernel_size=11)

        # Compute FFT
        fft_result = np.fft.fft(profile)
        n = len(profile)
        frequencies = np.fft.fftfreq(n)

        # Analyze only positive frequencies
        half_n = n // 2
        frequencies = frequencies[:half_n]
        magnitude = np.abs(fft_result[:half_n]) / n

        # Skip the DC component
        start_idx = max(1, int(0.001 * half_n))
        peaks, _ = find_peaks(magnitude[start_idx:], distance=5)
        peaks = peaks + start_idx

        if len(peaks) == 0:
            print(f"No peaks found in {orientation} FFT. Using evenly spaced divisions.")
            # Use evenly spaced divisions if no peaks found
            img_length = image.shape[1] if orientation == "vertical" else image.shape[0]
            edge_positions = np.linspace(0, img_length - 1, 6, dtype=int)
            return edge_positions, None

        # Find dominant frequency
        dominant_peak = peaks[np.argmax(magnitude[peaks])]
        dominant_freq = frequencies[dominant_peak]

        # Calculate estimated stripe width
        stripe_width = int(abs(1 / dominant_freq))

        # Sanity check on stripe width
        if stripe_width < 10 or stripe_width > min(image.shape) / 2:
            print(f"Warning: Unreasonable stripe width ({stripe_width}). Using evenly spaced divisions.")
            img_length = image.shape[1] if orientation == "vertical" else image.shape[0]
            edge_positions = np.linspace(0, img_length - 1, 6, dtype=int)
            return edge_positions, 1 / (img_length / 5)  # Approx frequency

        if self.debug:
            plt.figure(figsize=(12, 6))
            plt.subplot(211)
            plt.plot(profile)
            plt.title(f"{orientation.capitalize()} Intensity Profile")

            plt.subplot(212)
            plt.plot(frequencies, magnitude)
            plt.axvline(dominant_freq, color="r", linestyle="--")
            plt.title(f"FFT Magnitude Spectrum (Dominant period: {stripe_width} pixels)")
            plt.tight_layout()

            # Save the FFT analysis plot
            output_dir = os.path.dirname(self.output_path)
            fft_plot_path = os.path.join(output_dir, f"{orientation}_fft_analysis.png")
            plt.savefig(fft_plot_path)
            plt.close()

        # Calculate tile edge positions based on detected frequency
        img_length = image.shape[1] if orientation == "vertical" else image.shape[0]
        num_tiles = max(5, int(img_length / stripe_width))
        edge_positions = np.linspace(0, img_length - 1, num_tiles + 1, dtype=int)

        return edge_positions, dominant_freq

    def remove_stripes_vistamap(self, image_path, mask_path=None, output_path="process.tif"):
        """Main method implementing the VISTAmap approach"""
        self.output_path = output_path

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Preprocess image
        temp_image, mask, bright_mask, orig_normalized, (image_min, image_max) = self.load_and_preprocess(
            image_path, mask_path
        )

        if self.debug:
            plt.figure(figsize=(10, 10))
            plt.imshow(temp_image, cmap="gray")
            plt.title("Preprocessed Image")
            plt.colorbar()
            preprocessed_path = os.path.join(os.path.dirname(output_path), "preprocessed.png")
            plt.savefig(preprocessed_path)
            plt.close()

            plt.figure(figsize=(10, 10))
            plt.imshow(mask, cmap="gray")
            plt.title("Mask")
            plt.colorbar()
            mask_path = os.path.join(os.path.dirname(output_path), "mask.png")
            plt.savefig(mask_path)
            plt.close()

        # Detect stripes using FFT
        h_edge_positions, h_dominant_freq = self.detect_stripes_fft(
            temp_image, mask, orientation="horizontal"
        )
        v_edge_positions, v_dominant_freq = self.detect_stripes_fft(temp_image, mask, orientation="vertical")

        if self.debug and h_edge_positions is not None and v_edge_positions is not None:
            plt.figure(figsize=(10, 10))
            plt.imshow(temp_image, cmap="gray")
            for pos in v_edge_positions:
                plt.axvline(x=pos, color="r", linestyle="--")
            for pos in h_edge_positions:
                plt.axhline(y=pos, color="g", linestyle="--")
            plt.title("Detected Stripe Boundaries")
            boundaries_path = os.path.join(os.path.dirname(output_path), "stripe_boundaries.png")
            plt.savefig(boundaries_path)
            plt.close()

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

        if self.debug:
            plt.figure(figsize=(12, 8))
            plt.subplot(211)
            plt.plot(h_profile, label="Original Profile")
            plt.plot(h_lower, label="Lower Envelope")
            plt.plot(h_upper, label="Upper Envelope")
            plt.plot(h_ideal, label="Ideal Line")
            plt.legend()
            plt.title("Horizontal Intensity Profile and Envelopes")

            plt.subplot(212)
            plt.plot(h_multiplier, label="Horizontal Correction Factors")
            plt.axhline(1.0, color="r", linestyle="--")
            plt.legend()
            plt.title("Horizontal Correction Factors")
            plt.tight_layout()
            h_envelope_path = os.path.join(os.path.dirname(output_path), "horizontal_envelope.png")
            plt.savefig(h_envelope_path)
            plt.close()

        # Apply horizontal correction
        h_corrected = temp_image * h_multiplier[np.newaxis, :]

        if self.debug:
            plt.figure(figsize=(10, 10))
            plt.imshow(h_corrected, cmap="gray")
            plt.title("After Horizontal Correction")
            plt.colorbar()
            h_corrected_path = os.path.join(os.path.dirname(output_path), "horizontal_corrected.png")
            plt.savefig(h_corrected_path)
            plt.close()

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

        if self.debug:
            plt.figure(figsize=(12, 8))
            plt.subplot(211)
            plt.plot(v_profile, label="Original Profile")
            plt.plot(v_lower, label="Lower Envelope")
            plt.plot(v_upper, label="Upper Envelope")
            plt.plot(v_ideal, label="Ideal Line")
            plt.legend()
            plt.title("Vertical Intensity Profile and Envelopes")

            plt.subplot(212)
            plt.plot(v_multiplier, label="Vertical Correction Factors")
            plt.axhline(1.0, color="r", linestyle="--")
            plt.legend()
            plt.title("Vertical Correction Factors")
            plt.tight_layout()
            v_envelope_path = os.path.join(os.path.dirname(output_path), "vertical_envelope.png")
            plt.savefig(v_envelope_path)
            plt.close()

        # Apply vertical correction
        v_corrected = h_corrected * v_multiplier[:, np.newaxis]

        if self.debug:
            plt.figure(figsize=(10, 10))
            plt.imshow(v_corrected, cmap="gray")
            plt.title("After Vertical Correction")
            plt.colorbar()
            v_corrected_path = os.path.join(os.path.dirname(output_path), "vertical_corrected.png")
            plt.savefig(v_corrected_path)
            plt.close()

        # Apply final morphological operations if needed
        final_image = apply_final_morphological_operations(v_corrected, mask)

        if self.debug:
            plt.figure(figsize=(10, 10))
            plt.imshow(final_image, cmap="gray")
            plt.title("After Morphological Operations")
            plt.colorbar()
            morpho_path = os.path.join(os.path.dirname(output_path), "after_morphological.png")
            plt.savefig(morpho_path)
            plt.close()

        # Restore original bright pixels
        final_image[bright_mask] = orig_normalized[bright_mask]

        # Rescale to original intensity range
        final_image = final_image * (image_max - image_min) + image_min

        # Create comparison figure
        original = tifffile.imread(image_path)
        if original.ndim > 2:
            original = original.squeeze()

        plt.figure(figsize=(15, 10))
        plt.subplot(121)
        plt.imshow(original, cmap="gray")
        plt.title("Original Image")

        plt.subplot(122)
        plt.imshow(final_image, cmap="gray")
        plt.title("VISTAmap Corrected Image")

        # Save the comparison figure
        comparison_path = os.path.splitext(output_path)[0] + "_comparison.png"
        plt.savefig(comparison_path)
        plt.close()

        # Save profiles comparison
        h_profile_orig = np.nanmean(original * mask, axis=0)
        v_profile_orig = np.nanmean(original * mask, axis=1)

        h_profile_corr = np.nanmean(final_image * mask, axis=0)
        v_profile_corr = np.nanmean(final_image * mask, axis=1)

        plt.figure(figsize=(15, 10))
        plt.subplot(211)
        plt.plot(h_profile_orig, label="Original")
        plt.plot(h_profile_corr, label="Corrected")
        plt.title("Horizontal Intensity Profile")
        plt.legend()

        plt.subplot(212)
        plt.plot(v_profile_orig, label="Original")
        plt.plot(v_profile_corr, label="Corrected")
        plt.title("Vertical Intensity Profile")
        plt.legend()

        profile_path = os.path.splitext(output_path)[0] + "_profiles.png"
        plt.savefig(profile_path)
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
        help="Optional path to the mask file. If not provided, a mask will be generated automatically.",
    )
    parser.add_argument(
        "--output_path", type=str, default="output/process.tif", help="Path to save the output image."
    )
    parser.add_argument("--debug", default="False", help="Enable debug mode with diagnostic images.")
    parser.add_argument(
        "--tile_size", type=int, default=None, help="Specify the size of each tile (e.g., 256)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: Input image file not found: {args.image_path}")
        return

    if args.mask_path is not None and not os.path.exists(args.mask_path):
        print(f"Warning: Mask file not found: {args.mask_path}. Will generate automatically.")
        args.mask_path = None

    print(f"Starting VISTAmap processing on {args.image_path}")

    if args.tile_size is not None:
        print(f"Using specified tile size: {args.tile_size}")

    processor = VISTAmapDestriper(debug=args.debug)

    processor.remove_stripes_vistamap(args.image_path, args.mask_path, args.output_path)

    print("Processing completed successfully!")


if __name__ == "__main__":
    main()
