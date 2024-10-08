import argparse
import os

import cv2
import numpy as np
import tifffile
from scipy import ndimage, signal
from sklearn.linear_model import HuberRegressor, LinearRegression


class DeStripe:
    """Remove the stitching stripes from the WSI.
    Args:
        min_distance(int): Minimum distance between stripes. Default is 128
        correction_limit(float): Maximum correction factor. Default is 0.2
    """

    def __init__(self, min_distance=128, correction_limit=0.2):
        self.min_distance = min_distance
        self.correction_limit = correction_limit

    def load_and_preprocess(self, image_path, mask_path):
        assert os.path.exists(mask_path), f"Mask file not found: {mask_path}"
        original_image = tifffile.imread(image_path)
        mask = tifffile.imread(mask_path)

        if original_image.ndim > 2:
            original_image = original_image.squeeze()
        if mask.ndim > 2:
            mask = mask.squeeze()

        mask = (mask > 0).astype(np.uint8)

        image_min, image_max = np.percentile(original_image, (2, 98))
        normalized_image = np.clip((original_image - image_min) / (image_max - image_min), 0, 1)
        cropped_image = normalized_image * mask
        background_image = normalized_image * (1 - mask)

        return cropped_image, background_image, (image_min, image_max)

    def detect_stripes(self, background_image, orientation="vertical"):
        if orientation == "vertical":
            sobel_image = cv2.Sobel(background_image, cv2.CV_64F, 1, 0, ksize=3)
            line_sum = np.sum(sobel_image, axis=0)
        else:  # horizontal
            sobel_image = cv2.Sobel(background_image, cv2.CV_64F, 0, 1, ksize=3)
            line_sum = np.sum(sobel_image, axis=1)

        peaks, _ = signal.find_peaks(-line_sum, distance=self.min_distance)
        return peaks

    @staticmethod
    def adaptive_polyfit(x, y, degree=2, edge_threshold=0.1):
        x_norm = (x - x.min()) / (x.max() - x.min())

        central_mask = (x_norm > edge_threshold) & (x_norm < (1 - edge_threshold))
        if np.sum(central_mask) > degree + 1:
            X_central = np.column_stack([x[central_mask] ** i for i in range(degree + 1)])
            huber = HuberRegressor()
            huber.fit(X_central, y[central_mask])
            central_coeffs = huber.coef_[::-1]
        else:
            central_coeffs = np.polyfit(x[central_mask], y[central_mask], degree)

        left_mask = x_norm <= edge_threshold
        right_mask = x_norm >= (1 - edge_threshold)

        if np.sum(left_mask) > 1:
            left_reg = LinearRegression().fit(x[left_mask].reshape(-1, 1), y[left_mask])
            left_slope, left_intercept = left_reg.coef_[0], left_reg.intercept_
        else:
            left_slope, left_intercept = 0, y[0]

        if np.sum(right_mask) > 1:
            right_reg = LinearRegression().fit(x[right_mask].reshape(-1, 1), y[right_mask])
            right_slope, right_intercept = right_reg.coef_[0], right_reg.intercept_
        else:
            right_slope, right_intercept = 0, y[-1]

        def piecewise_fit(x):
            result = np.zeros_like(x, dtype=float)
            result[central_mask] = np.polyval(central_coeffs, x[central_mask])
            result[left_mask] = left_slope * x[left_mask] + left_intercept
            result[right_mask] = right_slope * x[right_mask] + right_intercept
            return result

        return piecewise_fit

    def process_image(self, image, vertical_minima, horizontal_minima):
        def correct_stripe(image, start, end, axis):
            if axis == 0:  # vertical stripe
                stripe = image[:, start:end]
                x = np.arange(start, end)
                y = np.nanmean(stripe, axis=0)
            else:
                stripe = image[start:end, :]
                x = np.arange(start, end)
                y = np.nanmean(stripe, axis=1)

            try:
                fit_func = self.adaptive_polyfit(x, y)
                fitted_y = fit_func(x)
                correction = np.where(fitted_y != 0, y / fitted_y, 1)
                correction = np.clip(correction, 1 - self.correction_limit, 1 + self.correction_limit)

                if axis == 0:
                    image[:, start:end] *= correction[np.newaxis, :]
                else:
                    image[start:end, :] *= correction[:, np.newaxis]
            except Exception as e:
                print(f"Warning: Failed to fit polynomial for stripe {start}:{end}. Error: {str(e)}")

        for i in range(len(vertical_minima) - 1):
            correct_stripe(image, vertical_minima[i], vertical_minima[i + 1], axis=0)

        for i in range(len(horizontal_minima) - 1):
            correct_stripe(image, horizontal_minima[i], horizontal_minima[i + 1], axis=1)

        # Global illumination correction
        x = np.arange(image.shape[1])
        y = np.nanmean(image, axis=0)
        try:
            fit_func = self.adaptive_polyfit(x, y)
            fitted_y = fit_func(x)
            correction = np.where(fitted_y != 0, y / fitted_y, 1)
            correction = np.clip(correction, 0.8, 1.2)  # Limit global correction
            image /= correction[np.newaxis, :]
        except Exception as e:
            print(f"Warning: Failed to fit polynomial for global illumination. Error: {str(e)}")

        image = ndimage.gaussian_filter(image, sigma=0.5)

        return image

    def remove_stripes(self, image_path, mask_path):
        try:
            cropped_image, background_image, (image_min, image_max) = self.load_and_preprocess(image_path, mask_path)

            v_minima = self.detect_stripes(background_image, orientation="vertical")
            h_minima = self.detect_stripes(background_image, orientation="horizontal")

            if len(v_minima) < 2 and len(h_minima) < 2:
                print(f"Warning: Not enough stripes detected in {image_path}")
                return cropped_image * (image_max - image_min) + image_min

            destriped_image = self.process_image(cropped_image, v_minima, h_minima)

            p_low, p_high = np.percentile(destriped_image, (2, 98))
            destriped_image = np.clip((destriped_image - p_low) / (p_high - p_low), 0, 1)

            return destriped_image * (image_max - image_min) + image_min
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None

    def process_single_image(self, image_path, mask_path):
        result = self.remove_stripes(image_path, mask_path)
        if result is not None:
            output_path = os.path.join("output", os.path.basename(image_path))
            tifffile.imwrite(output_path, result.astype(np.float32))
            print(f"Processed: {image_path}")
        else:
            print(f"Failed to process: {image_path}")


def main():
    parser = argparse.ArgumentParser(description="Remove stripes from a single image.")
    parser.add_argument("--image_path", type=str, default="protein_1.tif", help="Path to the image file.")
    parser.add_argument("--mask_path", type=str, default="protein_1_mask.tif", help="Path to the mask file.")
    parser.add_argument("--min_dist", type=int, default=300, help="Minimum distance between stripes.")
    parser.add_argument("--correction_limit", type=float, default=0.2, help="Maximum correction factor.")
    args = parser.parse_args()

    destriper = DeStripe(min_distance=args.min_dist, correction_limit=args.correction_limit)
    destriper.process_single_image(args.image_path, args.mask_path)


if __name__ == "__main__":
    main()
