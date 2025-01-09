REPOSITORY FOR INFERENCING SCRIPTS WITH ALL DATASETS AND ALL LOSS FUNCTIONS, WITH DIFFERENT AUGMENTATION TECHNIQUES

Summary of the Code:
Dataset Extraction: All datasets (LFW, CFP, JAFFE, CALFW, CPLFW) are extracted and loaded.
Augmentations: Applied augmentations like lighting variations, pose changes, and distortions to simulate challenging conditions.
Model Loading: Models trained with different loss functions (FuzzyArcFace, ArcFace, AdaptiveFace, etc.) are loaded for inference.
Embedding Extraction: For each dataset, embeddings are extracted from the original and augmented images.
Cosine Similarity Calculation: Cosine similarity between original and augmented embeddings is computed for each model.
Saving Results: Results are stored in a CSV file for analysis.

AUGMENTATION TECHNIQUES USED
1. Base Augmentation
Key: "Base"
Description: Applies general augmentations (e.g., random horizontal flips, brightness adjustments, random rotations). These modifications are mild and retain most of the original image characteristics.
Distortion Level: Minimal.
Purpose: Provides baseline augmentation to make the model robust to small variations.
2. Obfuscation
Keys: "Obfuscation_1", "Obfuscation_2"
Description: Applies Gaussian blur with varying levels of intensity.
"Obfuscation_1": Blurs the image with a lower blur radius (max_blur_radius=6, steps=6).
"Obfuscation_2": Applies more intense blurring (max_blur_radius=12, steps=12).
Distortion Level:
"Obfuscation_1": Mild to moderate distortion.
"Obfuscation_2": Higher distortion as details become less visible.
Purpose: Simulates scenarios where images are blurry, such as low-quality scans or motion blur.
3. Gaussian Noise
Keys: "Noise_1", "Noise_2", "Noise_3"
Description: Adds random Gaussian noise to images with varying intensity.
"Noise_1": Low noise level (std=5.0).
"Noise_2": Moderate noise level (std=25.0).
"Noise_3": High noise level (std=50.0).
Distortion Level:
"Noise_1": Minimal distortion.
"Noise_2": Moderate distortion.
"Noise_3": High distortion; fine details are overwhelmed by noise.
Purpose: Simulates noisy environments, such as images captured in poor lighting or using low-quality sensors.
4. Occlusion
Keys: "Occlusion_1", "Occlusion_2", "Occlusion_3", "Occlusion_4"
Description: Randomly occludes portions of the image by setting rectangular patches to zero (blackout).
"Occlusion_1": Small occlusion (max_hole_size=(30, 30)).
"Occlusion_2": Medium occlusion (max_hole_size=(60, 60)).
"Occlusion_3": Large occlusion (max_hole_size=(90, 90)).
"Occlusion_4": Maximum occlusion (max_hole_size=(112, 112)), equivalent to occluding the entire image.
Distortion Level:
"Occlusion_1": Low distortion; the occlusion covers a small area.
"Occlusion_4": Very high distortion; significant image information is lost.
Purpose: Simulates scenarios where parts of the image are obscured (e.g., objects blocking the view).
5. Compression
Keys: "Compression_1", "Compression_2"
Description: Simulates image compression artifacts by downsampling and upsampling with varying scales.
"Compression_1": High compression (scale_factors=[0.1, 0.5]), leading to significant loss of detail.
"Compression_2": Lower compression (scale_factors=[0.5, 0.75]), retaining more details.
Distortion Level:
"Compression_1": High distortion due to aggressive compression.
"Compression_2": Moderate distortion.
Purpose: Simulates low-quality image compression, such as JPEG artifacts or images resized to lower resolutions.
