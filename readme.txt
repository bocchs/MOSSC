This repository contains experimental images and code used in Bocchieri et al.
The images available here were detected to contain an interaction as described in the paper.
x-y_sensitivity.zip contains images captured in the x-y sensitivity experiment.
z_sensitivity.zip contains images captured in the z sensitivity experiment.
Images are stored as Numpy arrays in shapes of num_frames x 256 x 496.
Each Numpy file corresponds to frames captured for one gamma-ray source placement identified in the filename.
localize.py contains the denoising and localization algorithms described in the paper.

Instructions to run the code and reproduce results in the paper:
1. Download the zip files and code
2. Unzip the zip files
3. Comment/uncomment the desired image file in localize.py
4. Run localize.py

