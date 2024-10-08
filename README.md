# Stripe Removal for Optical Microscopy Images

This project provides a Python script for removing stitching stripes in optical microscopy images, specifically for SRS images. The script corrects the intensity values of the stripes to generate a seamless image with better visual quality.

## Comparison of Raw and Processed Images
![SRS DeStripe Comparison](https://github.com/Zhi-Li-SRS/SRS_DeStripe/blob/main/comparison/raw_vs_removed.png?raw=true)

- **Left Image**: Raw SRS (Stimulated Raman Scattering) image showing prominent stripes. These stripes are common in whole slide imaging in SRS microscopy.

- **Right Image**: The same SRS image after processing with our DeStripe algorithm. 

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/Zhi-Li-SRS/SRS_DeStripe.git
   cd SRS_DeStripe
   ```

2. Create a virtual environment (optional but recommended):
   ```
    conda create -n destripe python=3.9
    conda activate destripe

   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage
Prepare the input image and mask file. The mask file should be a binary image with the same dimensions as the input image, where the background is 0 and the foreground is 1. 

You can use the script in two ways:

### 1. Using command-line arguments

Run the script with the following command:

```
python de_stripe.py --image_path path/to/your/image.tif --mask_path path/to/your/mask.tif --min_dist 300 --correction_limit 0.2
```

Arguments:
- `--image_path`: Path to the input image file (default: "protein_1.tif")
- `--mask_path`: Path to the mask file (default: "protein_1_mask.tif")
- `--min_dist`: Minimum distance between stripes (default: 300). For example, if you know the patch size is 256, you can set it to 200.
- `--correction_limit`: Maximum correction factor (default: 0.2)

### 2. Modifying the script directly

You can also modify the `main()` function in the `de_stripe.py` file to set the paths and parameters directly:

```python
def main():
    remover = StripeRemover(min_distance=300, correction_limit=0.2)
    remover.process_single_image("path/to/your/image.tif", "path/to/your/mask.tif")

if __name__ == "__main__":
    main()
```

## Output

The processed image will be saved in the `output` directory with the same filename as the input image.

## Requirements
Recommand install the packages using conda:

The script requires the following Python packages:
- numpy
- opencv-python
- tifffile
- scipy
- scikit-learn

These can be installed using the `requirements.txt` file provided in the repository.

## License

This project is licensed under the MIT License - see the LICENSE file for details.