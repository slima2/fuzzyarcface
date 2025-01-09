REPOSITORY FOR INFERENCING SCRIPTS WITH ALL DATASETS AND ALL LOSS FUNCTIONS, WITH DIFFERENT AUGMENTATION TECHNIQUES

Summary of the Code:
Dataset Extraction: All datasets (LFW, CFP, JAFFE, CALFW, CPLFW) are extracted and loaded.
Augmentations: Applied augmentations like lighting variations, pose changes, and distortions to simulate challenging conditions.
Model Loading: Models trained with different loss functions (FuzzyArcFace, ArcFace, AdaptiveFace, etc.) are loaded for inference.
Embedding Extraction: For each dataset, embeddings are extracted from the original and augmented images.
Cosine Similarity Calculation: Cosine similarity between original and augmented embeddings is computed for each model.
Saving Results: Results are stored in a CSV file for analysis.
