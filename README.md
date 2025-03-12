#  Stripes Removal for Optical Microscopy Images

This project provides Python scripts (VISTAmap)for removing stitching stripes in optical microscopy images, specifically for SRS images. The scripts correct the intensity values of the stripes to generate a seamless image with better visual quality.

## Comparison of Raw and Processed Images
![SRS DeStripe Comparison](https://github.com/Zhi-Li-SRS/SRS_DeStripe/blob/main/comparison/raw_vs_removed.png?raw=true)

- **Left Image**: Raw SRS (Stimulated Raman Scattering) image showing prominent stripes. These stripes are common in whole slide imaging in SRS microscopy.

- **Right Image**: The same SRS image after processing with VISTAmap. 

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

VISTAmap Algorithm
The VISTAmap algorithm uses FFT analysis and adaptive morphological processing to remove stripes and correct vignetting in stitched tile images.

Run the script with the following command:

```
python vistamap.py --image_path path/to/your/image.tif --mask_path path/to/your/mask.tif --output_path output/result.tif --tile_size 256(optional)
```

Args
- `--image_path`: Path to the input image file (default: "791.tif")
- `--mask_path`: Path to the mask file (default: "srs_mask.tif")
- `--output_path`: Path to save the processed output image
- `--tile_size`: Optional arameter to specify the size of each tile (e.g., 256). If not provided, the algorithm will attempt to detect tile size using FFT analysis.


## Requirements
Recommand install the packages using conda:

The script requires the following Python packages:
- numpy
- opencv-python
- tifffile
- scipy
- scikit-learn

These can be installed using the `requirements.txt` file provided in the repository.

## Recommendations

- For optimal results with the VISTAmap algorithm, provide both the image and a binary mask separating foreground from background.
- If you know the exact tile size used in microscopy acquisition, specify it using the `--tile_size` parameter for faster process.
- VISTAmap generally produces better results for complex images with both striping and vignetting issues.


## License

This project is licensed under the MIT License - see the LICENSE file for details.



