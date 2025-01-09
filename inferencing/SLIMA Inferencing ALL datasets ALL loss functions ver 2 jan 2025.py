#!/usr/bin/env python
# coding: utf-8

# INFERENCING WITH ALL DATASETS AND ALL LOSS FUNCTIONS, WITH DIFFERENT AUGMENTATION TECHNIQUES

# Summary of the Code:
# Dataset Extraction: All datasets (LFW, CFP, JAFFE, CALFW, CPLFW) are extracted and loaded.
# Augmentations: Applied augmentations like lighting variations, pose changes, and distortions to simulate challenging conditions.
# Model Loading: Models trained with different loss functions (FuzzyArcFace, ArcFace, AdaptiveFace, etc.) are loaded for inference.
# Embedding Extraction: For each dataset, embeddings are extracted from the original and augmented images.
# Cosine Similarity Calculation: Cosine similarity between original and augmented embeddings is computed for each model.
# Saving Results: Results are stored in a CSV file for analysis.

# In[1]:


import os
import time
import logging
import random
import tarfile
import zipfile
#import cv2
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import numpy as np
from datetime import timedelta


# In[2]:


from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


# In[3]:


import torch.distributed as dist


# In[4]:


import torch.nn.functional as F
import kornia.augmentation as K
import kornia.filters as KF



# In[5]:


from torch.nn.parallel import DistributedDataParallel as DDP


# In[6]:


from collections import OrderedDict


# In[7]:


import torch.multiprocessing as mp


# In[8]:


import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler


# In[9]:


#from torch.nn import DataParallel


# In[10]:


from torch.utils.data import Subset
import random


# In[11]:


import concurrent.futures


# In[12]:


from concurrent.futures import ThreadPoolExecutor


# In[13]:


from torchvision.models import resnet101, ResNet101_Weights


# In[14]:


import io


# In[15]:


from torchvision.datasets import ImageFolder
from torchvision import models


from torchvision.transforms import ToPILImage, ToTensor
from PIL import ImageFilter
import torch.nn.functional as F


# In[16]:


import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score


# In[17]:


# Disable semaphore tracking to avoid resource tracker warnings
from multiprocessing.resource_tracker import ResourceTracker


# In[18]:


from torch.utils.data import Dataset


# In[19]:


#import multiprocessing as mp
def _fix_semaphore_tracker():
    ResourceTracker._CLEANUP_FUNCS.pop("semaphore", None)


# In[20]:


base_path = '/home/rapids/notebooks/slima/DATABASES'
model_save_path = '/home/rapids/notebooks/slima'
subdirectory = 'pth'

# Set output directory
output_dir = '/home/rapids/notebooks/slima'

#num classes lfw
num_classes=5749
batchsize=32
numworkers=4
percentilevalue=10
#thresholdtype="percentile"
thresholdtype="fixed"
depthextraction=0
subsetratio=1


# In[21]:


world_size = torch.cuda.device_count()


# In[22]:


if torch.cuda.device_count() > 1:
    print(f"{torch.cuda.device_count()} GPUs available")
else:
    print("Single GPU or no GPU available")


# In[23]:


# Check if GPU is available and move the model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# In[24]:


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# In[25]:


# If using CUDA (PyTorch)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)  # if you are using multi-GPU.


# In[26]:


mp.set_start_method('forkserver', force=True)


# In[27]:


#_fix_semaphore_tracker()


# In[28]:


# Step 2: Set up available GPUs and seed for reproducibility
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_gpus = torch.cuda.device_count()

def set_seed(seed=42):
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed()


# In[29]:


# Setup logging
logging.basicConfig(filename='processing_log42.log', level=logging.INFO, 
                    format='%(asctime)s %(message)s', filemode='w')


# In[30]:


def log_progress(dataset_name, progress_percentage, images_processed, total_images):
    logging.info(f"{dataset_name}: {progress_percentage:.2f}% complete ({images_processed}/{total_images} images)")


# In[31]:


### ----------------- Data Extraction -------------------

# Paths to the archives and where to extract them
lfw_tgz_path = os.path.join(base_path, 'lfw.tgz')
lfw_extract_path = os.path.join(base_path, 'extracted', 'lfw/lfw')

cfp_tar_path = os.path.join(base_path, 'CFP.tar')
cfp_extract_path = os.path.join(base_path, 'extracted', 'Data/Images')

jaffedbase_tar_path = os.path.join(base_path, 'jaffedbase.tar')
jaffedbase_extract_path = os.path.join(base_path, 'extracted', 'jaffedbase/jaffedbase')

calfw_zip_path = os.path.join(base_path, 'calfw.zip')
calfw_extract_path = os.path.join(base_path, 'extracted', 'calfw/calfw')

cplfw_zip_path = os.path.join(base_path, 'cplfw.zip')
cplfw_extract_path = os.path.join(base_path, 'extracted', 'cplfw/cplfw')


# In[32]:


# Extract tar files (CFP, Jaffedbase, LFW)
for tar_path, extract_path in [(cfp_tar_path, cfp_extract_path), 
                               (jaffedbase_tar_path, jaffedbase_extract_path),
                               (lfw_tgz_path, lfw_extract_path)]:
    if not os.path.exists(extract_path):
        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(extract_path)



# In[33]:


# Extract zip files (CALFW, CPLFW)
for zip_path, extract_path in [(calfw_zip_path, calfw_extract_path), 
                               (cplfw_zip_path, cplfw_extract_path)]:
    if not os.path.exists(extract_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)



# In[34]:


class FlatDirectoryImageDataset(Dataset):
    """
    Custom Dataset for loading images from a directory where images
    are stored in the same folder, with class labels extracted from filenames.
    """
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        
        # Walk through the dataset directory and gather images and labels
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png', '.tiff')):
                    img_path = os.path.join(root, file)
                    self.image_paths.append(img_path)
                    
                    # Extract class label from the filename (before the first underscore)
                    class_name = file.split('_')[0]
                    
                    # If the class hasn't been seen before, add it to class_to_idx mapping
                    if class_name not in self.class_to_idx:
                        self.class_to_idx[class_name] = len(self.class_to_idx)
                    
                    # Assign the numerical label based on class_to_idx mapping
                    label = self.class_to_idx[class_name]
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label


# In[35]:


# Step 6: Define the iResNet100 architecture
class iResNet100(nn.Module):
    def __init__(self, num_classes=num_classes):  # LFW classes
        super(iResNet100, self).__init__()
        #self.model = models.resnet101(pretrained=True)
        self.model = resnet101(weights=ResNet101_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)


# In[36]:


class GPUAugmentations(nn.Module):
    """
    A module that applies multiple GPU-accelerated augmentations to a batch of images.
    Images are expected to be tensors of shape (B, C, H, W) on CUDA.
    """
    def __init__(self, brightness=0.5, rotation_degrees=30):
        super().__init__()
        self.augmentations = nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            K.ColorJitter(brightness=brightness),
            K.RandomRotation(degrees=rotation_degrees)
        )

    def forward(self, x):
        # x: (B, C, H, W)
        return self.augmentations(x)

class RandomOcclusionGPU(nn.Module):
    """
    Randomly occludes portions of the image by setting random patches to zero.
    """
    def __init__(self, max_holes=3, max_hole_size=(30, 30)):
        super().__init__()
        self.max_holes = max_holes
        self.max_hole_size = max_hole_size

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        for i in range(B):
            for _ in range(self.max_holes):
                hole_w = random.randint(1, self.max_hole_size[0])
                hole_h = random.randint(1, self.max_hole_size[1])
                top = random.randint(0, H - hole_h)
                left = random.randint(0, W - hole_w)
                x[i, :, top:top+hole_h, left:left+hole_w] = 0
        return x

class AddGaussianNoiseGPU(nn.Module):
    def __init__(self, mean=0., std=25.):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        # x: (B, C, H, W)
        noise = torch.normal(mean=self.mean, std=self.std, size=x.size(), device=x.device)
        # Ensure noise is scaled to [-1,1] if working with normalized images
        noise = noise / 255.0
        return torch.clamp(x + noise, -1, 1)  # Assuming images normalized to [-1,1]

class CompressImageGPU(nn.Module):
    """
    Approximate JPEG compression artifacts on GPU by:
    - Downsampling and upsampling the image (simulating loss of detail).
    - Optionally adding some block noise.
    This is a heuristic, not a true JPEG compression.
    """
    def __init__(self, scale_factors=[0.5, 0.75], block_noise_prob=0.5):
        super().__init__()
        self.scale_factors = scale_factors
        self.block_noise_prob = block_noise_prob

    def forward(self, x):
        B, C, H, W = x.shape
        # Randomly choose a scale factor to simulate compression
        scale_factor = random.choice(self.scale_factors)
        new_H = int(H * scale_factor)
        new_W = int(W * scale_factor)
        
        # Downsample
        x_ds = F.interpolate(x, size=(new_H, new_W), mode='bilinear', align_corners=False)
        # Upsample back
        x_rec = F.interpolate(x_ds, size=(H, W), mode='bilinear', align_corners=False)

        # Optionally add block noise
        if random.random() < self.block_noise_prob:
            # Add block patterns
            block_size = 8
            for i in range(B):
                # Create block noise pattern
                for row in range(0, H, block_size):
                    for col in range(0, W, block_size):
                        # Slight random shift for block value
                        noise_val = (torch.rand(1, device=x.device)-0.5)/50
                        x_rec[i, :, row:row+block_size, col:col+block_size] += noise_val
            x_rec = torch.clamp(x_rec, -1, 1)

        return x_rec

class ObfuscateImageGPU(nn.Module):
    """
    Apply Gaussian blur multiple times with increasing radius to simulate different levels of obfuscation.
    Kornia's gaussian_blur2d expects kernel sizes and sigma. We simulate increasing blur by increasing sigma.
    """
    def __init__(self, max_blur_radius=6, steps=6):
        super().__init__()
        self.steps = steps
        self.sigmas = [((i+1)*(max_blur_radius/self.steps)) for i in range(self.steps)]

    def forward(self, x):
        # Returns a list of blurred versions, but to integrate with the original code,
        # we'll return them as a list of tensors. The calling code can stack them if needed.
        B, C, H, W = x.shape
        blurred_images = []
        for sigma in self.sigmas:
            # Kernel size approx: kernel_size ~ (sigma * 3)
            k = max(3, int(sigma*3)//2*2+1)  # Ensure odd kernel size
            blurred = KF.gaussian_blur2d(x, (k, k), (sigma, sigma))
            blurred_images.append(blurred)
        return blurred_images


# 1. Base Augmentation
# Key: "Base"
# Description: Applies general augmentations (e.g., random horizontal flips, brightness adjustments, random rotations). These modifications are mild and retain most of the original image characteristics.
# Distortion Level: Minimal.
# Purpose: Provides baseline augmentation to make the model robust to small variations.
# 2. Obfuscation
# Keys: "Obfuscation_1", "Obfuscation_2"
# Description: Applies Gaussian blur with varying levels of intensity.
# "Obfuscation_1": Blurs the image with a lower blur radius (max_blur_radius=6, steps=6).
# "Obfuscation_2": Applies more intense blurring (max_blur_radius=12, steps=12).
# Distortion Level:
# "Obfuscation_1": Mild to moderate distortion.
# "Obfuscation_2": Higher distortion as details become less visible.
# Purpose: Simulates scenarios where images are blurry, such as low-quality scans or motion blur.
# 3. Gaussian Noise
# Keys: "Noise_1", "Noise_2", "Noise_3"
# Description: Adds random Gaussian noise to images with varying intensity.
# "Noise_1": Low noise level (std=5.0).
# "Noise_2": Moderate noise level (std=25.0).
# "Noise_3": High noise level (std=50.0).
# Distortion Level:
# "Noise_1": Minimal distortion.
# "Noise_2": Moderate distortion.
# "Noise_3": High distortion; fine details are overwhelmed by noise.
# Purpose: Simulates noisy environments, such as images captured in poor lighting or using low-quality sensors.
# 4. Occlusion
# Keys: "Occlusion_1", "Occlusion_2", "Occlusion_3", "Occlusion_4"
# Description: Randomly occludes portions of the image by setting rectangular patches to zero (blackout).
# "Occlusion_1": Small occlusion (max_hole_size=(30, 30)).
# "Occlusion_2": Medium occlusion (max_hole_size=(60, 60)).
# "Occlusion_3": Large occlusion (max_hole_size=(90, 90)).
# "Occlusion_4": Maximum occlusion (max_hole_size=(112, 112)), equivalent to occluding the entire image.
# Distortion Level:
# "Occlusion_1": Low distortion; the occlusion covers a small area.
# "Occlusion_4": Very high distortion; significant image information is lost.
# Purpose: Simulates scenarios where parts of the image are obscured (e.g., objects blocking the view).
# 5. Compression
# Keys: "Compression_1", "Compression_2"
# Description: Simulates image compression artifacts by downsampling and upsampling with varying scales.
# "Compression_1": High compression (scale_factors=[0.1, 0.5]), leading to significant loss of detail.
# "Compression_2": Lower compression (scale_factors=[0.5, 0.75]), retaining more details.
# Distortion Level:
# "Compression_1": High distortion due to aggressive compression.
# "Compression_2": Moderate distortion.
# Purpose: Simulates low-quality image compression, such as JPEG artifacts or images resized to lower resolutions.

# In[37]:


def apply_all_transformations_gpu(images=None, return_keys_only=False):
    """
    Returns augmented image batches with labels identifying each augmentation
    or just the augmentation keys if `return_keys_only` is True.
    """
    augmentations = {
        "Base": GPUAugmentations()(images) if images is not None else None,
        "Obfuscation_1": ObfuscateImageGPU(max_blur_radius=6, steps=6)(images)[0] if images is not None else None,
        "Obfuscation_2": ObfuscateImageGPU(max_blur_radius=12, steps=12)(images)[1] if images is not None else None,
        "Noise_1": AddGaussianNoiseGPU(mean=0., std=5.)(images) if images is not None else None,
        "Noise_2": AddGaussianNoiseGPU(mean=0., std=25.)(images) if images is not None else None,
        "Noise_3": AddGaussianNoiseGPU(mean=0., std=50.)(images) if images is not None else None,
        "Occlusion_1": RandomOcclusionGPU(max_holes=1, max_hole_size=(30, 30))(images.clone()) if images is not None else None,
        "Occlusion_2": RandomOcclusionGPU(max_holes=1, max_hole_size=(60, 60))(images.clone()) if images is not None else None,
        "Occlusion_3": RandomOcclusionGPU(max_holes=1, max_hole_size=(90, 90))(images.clone()) if images is not None else None,
        "Occlusion_4": RandomOcclusionGPU(max_holes=1, max_hole_size=(112, 112))(images.clone()) if images is not None else None,
        "Compression_1": CompressImageGPU(scale_factors=[0.1, 0.5], block_noise_prob=0.5)(images) if images is not None else None,
        "Compression_2": CompressImageGPU(scale_factors=[0.5, 0.75], block_noise_prob=0.5)(images) if images is not None else None,
    }
    return list(augmentations.keys()) if return_keys_only else augmentations


# In[38]:


expected_augmentation_keys = apply_all_transformations_gpu(return_keys_only=True)


# In[39]:


num_transformations = len(apply_all_transformations_gpu(return_keys_only=True))
print(f"Number of Transformations: {num_transformations}")


# In[40]:


#for balancing out augmentations and negative sampled images per batch
sample_pairs_negative=batchsize*num_transformations  #max value
sample_pairs_positive=batchsize   #max value


# In[41]:


def remove_module_prefix(state_dict):
    """Remove 'module.' prefix from keys in state_dict if model was trained with DataParallel."""
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k[7:] if k.startswith("module.") else k  # Remove 'module.' prefix if present
        new_state_dict[new_key] = v
    return new_state_dict


# In[42]:


def load_model(model, rank):
    """
    Load a pre-initialized model into the appropriate GPU and wrap in DDP.
    """
    device = torch.device(f"cuda:{rank}")
    model = model.to(device)  # Move model to the correct rank-specific GPU

    # Wrap the model in DistributedDataParallel
    model = DDP(model, device_ids=[rank], output_device=rank)
    
    return model


# In[43]:


# Model paths
model_filenames = [
    'FuzzyArcFace_2024-10-27.pth', 
    'FuzzyArcFace_tau 0.1.pth', 
    'FuzzyArcFace_tau 0.5.pth', 
    'ArcFace_2024-10-28.pth', 
    'AdaptiveFace_2024-10-29.pth',
    'VPL_2024-10-29.pth', 
    'SphereFace2_2024-10-30.pth', 
    'UniFace_2024-10-31.pth'
]


# In[44]:


# Define model names in the same order as model_filenames
model_names = ["FuzzyArcFace9", 
               "FuzzyArcFace5",
               "FuzzyArcFace1",
               "ArcFace", 
               "AdaptiveFace", 
               "VPL", 
               "SphereFace2", 
               "UniFace"
              ]


# In[45]:


transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


# In[46]:


# Function to extract embeddings
def extract_embedding(model, image, device):
    model.eval()
    
    # Check if the input is already batched (4D)
    if image.dim() == 3:  # If the image is 3D, add the batch dimension
        image = image.unsqueeze(0)
    
    image = image.to(device)
    
    with torch.no_grad():
        embedding = model(image)  # No need to unsqueeze again here
    return embedding


# In[47]:


def get_subset_loader(dataloader, subset_ratio=subsetratio, batch_size=batchsize):
    """
    Get a DataLoader with a random subset specific to the rank.
    """
    dataset_size = len(dataloader.dataset)
    subset_size = max(1, int(subset_ratio * dataset_size))  # Ensure at least one sample

    # Use DistributedSampler for rank-specific subsets
    sampler = DistributedSampler(dataloader.dataset, num_replicas=dataloader.sampler.num_replicas, rank=dataloader.sampler.rank)
    subset_indices = list(iter(sampler))[:subset_size]  # Get only the rank-specific subset
    subset_dataset = Subset(dataloader.dataset, subset_indices)

    return DataLoader(subset_dataset, batch_size=batch_size, shuffle=False, num_workers=numworkers)


# In[48]:


def process_batch(model, images, device):
    """
    Process a batch of images and return embeddings for the original images and their augmentations.
    
    Arguments:
    - model: The model used for extracting embeddings.
    - images: A batch of images (tensor).
    - device: Device on which to run the model (e.g., 'cuda' or 'cpu').
    
    Returns:
    - original_embeddings: A dictionary with original embeddings for each image in the batch.
    - augmented_embeddings: A dictionary with augmented embeddings for each image in the batch.
    """
    model.eval()
    start_time = time.time()  # Start timing
   

    original_embeddings = {}
    augmented_embeddings = {}

    # images is already a batch of shape (N, C, H, W)
    # Ensure images are on GPU
    images = images.to(device)

    with torch.no_grad():
        # Extract original embeddings
        original_batch_embedding = model(images)  # (N, embedding_dim)

    # Process each image in the batch
    for i in range(images.size(0)):
        # Store original embeddings in a dictionary with keys as indices
        print(f"Processing image {i + 1} of {images.size(0)}")
        original_embeddings[i] = original_batch_embedding[i].cpu()
        
        # Apply augmentations in parallel (runs on CPU threads)
        aug_start_time = time.time()
        all_augs = apply_all_transformations_gpu(images)
        
        # We’ll process all augmentations in batches for efficiency:
        #augmented_embeddings = {i: [] for i in range(images.size(0))}
        augmented_embeddings = {i: {} for i in range(images.size(0))}
        
  
        with torch.no_grad():
        # For each augmented variant, run through the model
            for aug_type, aug_batch in all_augs.items():
                aug_emb_batch = model(aug_batch)  # (N, embedding_dim)
                aug_emb_batch = aug_emb_batch.cpu()
                # Append each image's augmented embedding
                for i in range(images.size(0)):
                    if aug_type not in augmented_embeddings[i]:
                        augmented_embeddings[i][aug_type] = []  # Initialize as a list
                    augmented_embeddings[i][aug_type].append(aug_emb_batch[i])
                    #print(f"Augmentations took {time.time() - aug_start_time:.2f} seconds")
            
    print(f"Total batch processing took {time.time() - start_time:.2f} seconds")
    return original_embeddings, augmented_embeddings
  


# In[49]:


def generate_pairs(images, labels):
    """
    Generate positive and negative pairs from a batch of images and labels.
    """
    positive_pairs = []
    negative_pairs = []

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if labels[i] == labels[j]:  # Positive pair
                positive_pairs.append((i, j, 1))
            else:  # Negative pair
                negative_pairs.append((i, j, 0))

    return positive_pairs, negative_pairs


# In[50]:


def process_single_dataset(
    dataloader,
    models_to_test,
    output_dir,
    dataset_name,
    loss_function_name,  # New parameter for loss function
    subset_ratio=subsetratio,
    batch_size=batchsize,
    threshold_type=thresholdtype,
    percentile_value=percentilevalue,
    rank=0,
    device=device
):

    results_df = [pd.DataFrame() for _ in range(rank+1)]  # One DataFrame for each rank
    # Initialize dictionaries for embeddings storage and metrics calculation
    all_original_embeddings = {}
    all_augmented_embeddings = {}
    similarities = []
    true_labels = []
    total_images = len(dataloader.dataset)  # Use dataloader instead of subset_loader
    images_processed = 0
    print(f"Total images: {total_images}")
    # Initialize augmented_metrics globally
    augmented_metrics = {aug_type: {"similarities": [], "true_labels": []} for aug_type in expected_augmentation_keys}

    for batch_idx, (images, labels) in enumerate(dataloader):
        print(f"Rank {rank}: Batch {batch_idx} - Image shape: {images.shape}, Labels: {labels}")
        images = images.to(device)
        batch_size = images.size(0)
        print(f"Batch size: {batch_size}")
        print(f"Rank {rank}: Processing batch {batch_idx} on device {images.device}")
        for model_name, model in models_to_test.items():
            # Add short name for debugging
            model_display_name = getattr(model, "short_name", model_name)  # Use short_name if available
            print(f"Processing batch with model: {model_display_name}")
            assert all(p.device == device for p in model.parameters()), f"Model {model_display_name} has parameters on the wrong device!"
              # Initialize original_embedding in case of errors in the batch
            original_embedding = {}
            try:
                #generate augmented images per each original image in the batch
                original_embedding, augmented_embedding = process_batch(model, images, device)

                if len(original_embedding) != len(labels):
                    print(f"Warning: original_embedding length ({len(original_embedding)}) does not match labels length ({len(labels)})")
                    continue

                # Compare augmented images against its original image. True labels are true always 
                for i in range(len(original_embedding)):
                    emb_i = extract_tensor_from_nested_dict(original_embedding[i])
                    for aug_type, augmented_emb in augmented_embedding[i].items():
                        emb_k = extract_tensor_from_nested_dict(augmented_emb)
                        if emb_k.dim() > 1:
                            emb_k = emb_k.flatten()  # Ensure it's a 1D vector
                        similarity = F.cosine_similarity(emb_i.unsqueeze(0), emb_k.unsqueeze(0)).item()
                        similarities.append(similarity)
                        true_labels.append(1)  # Positive pairs
                        # Append to augmentation-specific metrics
                        augmented_metrics[aug_type]["similarities"].append(similarity)
                        augmented_metrics[aug_type]["true_labels"].append(1)

                # Generate positive and negative pairs
                positive_pairs, negative_pairs = generate_pairs(images, labels)
                

                # Limit the number of pairs to 4 each (adjust if needed)
                num_positive_samples = min(sample_pairs_positive, len(positive_pairs)) #in general positive samples are less than negative
                num_negative_samples = min(sample_pairs_negative, len(negative_pairs))
                print(f"AFTER Num positive samples...{num_positive_samples}, Num negative samples...{num_negative_samples}")

                # Randomly sample pairs
                sampled_positive_pairs = random.sample(positive_pairs, num_positive_samples) if positive_pairs else []
                sampled_negative_pairs = random.sample(negative_pairs, num_negative_samples) if negative_pairs else []
                print(f"Sampled positive pairs...{sampled_positive_pairs}, Sampled negative pairs...{sampled_negative_pairs}")
                
                # Compute similarities for negative pairs. True labels are false always
                for i, j, label in sampled_negative_pairs:
                    emb_i = extract_tensor_from_nested_dict(original_embedding[i])
                    emb_j = extract_tensor_from_nested_dict(original_embedding[j])
    
                    if emb_i is None or emb_j is None:
                        print(f"Skipping comparison for indices {i}, {j} due to incompatible types.")
                        continue
    
                    if emb_i.dim() > 1:
                        emb_i = emb_i.flatten()
                    if emb_j.dim() > 1:
                        emb_j = emb_j.flatten()
    
                    similarity = F.cosine_similarity(emb_i.unsqueeze(0), emb_j.unsqueeze(0)).item()
                    similarities.append(similarity)
                    true_labels.append(0)

                # Compute similarities for positive pairs. True labels are true always, if there are positive samples.
                for i, j, label in sampled_positive_pairs:
                    emb_i = extract_tensor_from_nested_dict(original_embedding[i])
                    emb_j = extract_tensor_from_nested_dict(original_embedding[j])

    
                    if emb_i is None or emb_j is None:
                        print(f"Skipping comparison for indices {i}, {j} due to incompatible types.")
                        continue
    
                    if emb_i.dim() > 1:
                        emb_i = emb_i.flatten()
                    if emb_j.dim() > 1:
                        emb_j = emb_j.flatten()

                    #similarity for positive pair
                    similarity = F.cosine_similarity(emb_i.unsqueeze(0), emb_j.unsqueeze(0)).item()
                    similarities.append(similarity)
                    true_labels.append(1)
            except Exception as exc:
                print(f"Error processing batch {batch_idx} for model {model_name}: {exc}")
                
        # Store batch results for this dataset, ensuring original_embedding is available and consistent
        if original_embedding and len(original_embedding) == len(labels):  # Add this condition
            batch_results = {
                'Index': list(original_embedding.keys()),
                'True Label': labels.cpu().numpy(),
                'Embedding': [embedding.numpy() for embedding in original_embedding.values()]
            }

            batch_df = pd.DataFrame(batch_results)
            #results_df = pd.concat([results_df, batch_df], ignore_index=True)
            results_df[rank] = pd.concat([results_df[rank], batch_df], ignore_index=True)

            # intermediate_output_path = f"{output_dir}/{dataset_name}_{model_name}_partial_results17.csv"
            intermediate_output_path = f"{output_dir}/{dataset_name}_{model_display_name}_partial_results42_rank{rank}.csv"
            results_df[rank].to_csv(intermediate_output_path, mode='w', header=True, index=False)
 
            logging.info(f"Intermediate results for dataset '{dataset_name}' (rank {rank}) saved to {intermediate_output_path}.")
            logging.info(f"Rank {rank}: Processed {len(results_df[rank])} rows in total")
        else:
            print(f"No valid embeddings for batch {batch_idx} or mismatch in original_embedding and labels.")

        images_processed += batch_size
        progress_percentage = (images_processed / total_images) * 100
        log_progress(dataset_name, progress_percentage, images_processed, total_images)


    #if similarities.size == 0:
    if len(similarities) == 0:
        raise ValueError("Similarities array is empty. Ensure data pipeline is functioning correctly.")

    # Calculate metrics based on similarities and true labels
    if threshold_type == "percentile":
        threshold = np.percentile(similarities, percentile_value)
        print(f"Threshold used for predictions: {threshold}")
    elif threshold_type == "mean":
        threshold = np.mean(similarities)
    elif threshold_type == "median":
        threshold = np.median(similarities)
    elif threshold_type == "fixed":
        threshold = 0.5
    else:
        threshold = 0.5  # Default threshold if none is specified

    predictions = (np.array(similarities) >= threshold).astype(int)
    accuracy = np.mean(predictions == np.array(true_labels))

    print("\nDetailed Results:")
    for idx in range(len(similarities)):
        print(f"Similarity: {similarities[idx]:.4f}, True Label: {true_labels[idx]}, Prediction: {predictions[idx]}, {'Correct' if predictions[idx] == true_labels[idx] else 'Incorrect'}")

    print(f"\nTotal comparisons: {len(similarities)}")
    print(f"Final Accuracy: {accuracy:.4f}")

    print(f"\nOVERALL Metrics per dataset and rank: {loss_function_name}, Dataset: {dataset_name}, Rank: {rank}:")
    # Proportion of similarities above threshold
    proportion_above_threshold = np.sum(np.array(similarities) >= threshold) / len(similarities)
    # Recall
    recall = recall_score(true_labels, predictions, zero_division=0)
    # Precision
    precision = precision_score(true_labels, predictions, zero_division=0)
    # F1 Score
    f1 = f1_score(true_labels, predictions, zero_division=0)
    # Mean similarity
    mean_similarity = np.mean(similarities)

    # Calculate ROC-AUC
    if len(np.unique(true_labels)) > 1:
        fpr, tpr, _ = roc_curve(true_labels, similarities)
        roc_auc = roc_auc_score(true_labels, similarities)
    else:
        fpr, tpr, roc_auc = [], [], 0

    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Proportion Above Threshold: {proportion_above_threshold:.4f}")
    print(f"Mean Similarity: {mean_similarity:.4f}")

    print(f"\nMetrics per augmentation type for Loss Function: {loss_function_name}, Dataset: {dataset_name}, Rank: {rank}:")
    for aug_type, metrics in augmented_metrics.items():
        if len(metrics["true_labels"]) > 0:  # Ensure data exists for this augmentation
            num_augmentations = len(metrics["true_labels"])
            aug_predictions = [1 if sim >= threshold else 0 for sim in metrics["similarities"]]
            # Calculate recall
            aug_recall = recall_score(metrics["true_labels"], aug_predictions, zero_division=0)
            # Proportion of predictions above threshold
            proportion_above_threshold = sum(sim >= threshold for sim in metrics["similarities"]) / len(metrics["similarities"])
            # Mean similarity for additional insight
            mean_similarity = np.mean(metrics["similarities"])
            print(f"Augmentation: {aug_type}, Recall: {aug_recall:.4f}, Proportion Above Threshold: {proportion_above_threshold:.4f}, Mean Similarity: {mean_similarity:.4f}, Number of Augmentations: {num_augmentations}")

    # Save final results for this dataset
    # final_output_path = f"{output_dir}/{dataset_name}_{model_display_name}_results42.csv"
    # results_df[rank].to_csv(final_output_path, mode='w', header=True, index=False)
    #logging.info(f"Processing complete for dataset '{dataset_name}'. Results saved to {final_output_path}.")
    
    return results_df[rank], accuracy, roc_auc, fpr, tpr


# In[51]:


def extract_tensor_from_nested_dict(data, depth=depthextraction):
    """
    Recursively extract the first tensor-compatible element from a deeply nested structure
    and flatten it if necessary, with added debugging.
    """
    indent = "  " * depth  # Indentation for readability in debug output
    #print(f"{indent}Data type at depth {depth}: {type(data)}")  # Print data type

    if isinstance(data, torch.Tensor):
        #print(f"{indent}Found tensor at depth {depth}. Shape: {data.shape}")
        return data.flatten()  # Flatten tensor if it’s not 1-dimensional

    elif isinstance(data, np.ndarray):
        #print(f"{indent}Found numpy array at depth {depth}. Shape: {data.shape}")
        return torch.tensor(data).flatten()  # Convert numpy arrays directly and flatten

    elif isinstance(data, list):
        #print(f"{indent}Exploring list at depth {depth}, length: {len(data)}")
        for item in data:
            tensor_value = extract_tensor_from_nested_dict(item, depth + 1)
            if tensor_value is not None:
                return tensor_value

    elif isinstance(data, dict):
        #print(f"{indent}Exploring dict at depth {depth}, keys: {list(data.keys())}")
        for key, value in data.items():
            tensor_value = extract_tensor_from_nested_dict(value, depth + 1)
            if tensor_value is not None:
                return tensor_value

    # Explicitly state when None is returned at a level where no tensor was found
    print(f"{indent}No tensor found at depth {depth} in current structure: {data}")
    return None  # Return None if no tensor-compatible data is found


# In[52]:


# Configure logging (you can adjust the level as needed)
logging.basicConfig(level=logging.INFO)  # Set to DEBUG for detailed prints, INFO for less


# In[53]:


import matplotlib.pyplot as plt

# def plot_roc_curve(fpr, tpr, roc_auc, model_name):
#     plt.figure()
#     plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})')
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title(f'ROC Curve for {model_name}')
#     plt.legend(loc="lower right")
#     plt.show()


# In[54]:


def run_tests(models_to_test, test_loaders, output_dir, subset_ratios, batch_size=batchsize, rank=0):
    """
    Runs tests on the specified models using the provided test loaders.
    """
    for test_name, test_loader in test_loaders.items():
        print(f"Rank {rank}: Testing on {test_name}...")
        sampler = test_loader.sampler
        print(f"Rank {rank}: Total images in dataset: {len(sampler)}")  # Log sampler size
        
        # Use the subset_ratio to get a subset loader
        subset_loader = get_subset_loader(test_loader, subset_ratio=subset_ratios.get(test_name, subsetratio), batch_size=batch_size)
        device = torch.device(f"cuda:{rank}")

        # Evaluate each model on the subset
        for model_name, model in models_to_test.items():
            model_display_name = getattr(model, "short_name", model_name)
            print(f"Rank {rank}: Evaluating model '{model_display_name}' on dataset '{test_name}'...")


            results_df, accuracy, roc_auc, fpr, tpr = process_single_dataset(
                subset_loader,
                {model_name: model},
                output_dir,
                test_name,
                model_display_name,  # Pass loss function name
                subset_ratio=subsetratio,
                rank=rank,
                device=device
            )

            print(f"Rank {rank}: Dataset: {test_name}, Model: {model_display_name}, Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")
            #plot_roc_curve(fpr, tpr, roc_auc, f"{model_name}_{test_name}")


# In[55]:


def initialize_models(model_filepaths, rank, world_size):
    """
    Initialize and load models using DistributedDataParallel with rank-specific optimizations.
    """
    models_to_test = {}

    for model_name, model_path in model_filepaths.items():
        temp_model_path = f"{model_name}_temp.pth"
        
        # Check if the temp file exists
        if os.path.exists(temp_model_path):
            # Load the model state from the existing temp file
            print(f"Rank {rank}: Found existing temp file for {model_name}. Loading model state...")
            model = iResNet100()
            device = torch.device(f"cuda:{rank}")
            model = model.to(device)
            model.load_state_dict(torch.load(temp_model_path, map_location=device))
        else:
            if rank == 0:
                # Load and save the model on rank 0
                print(f"Rank 0: Saving temp file for {model_name}...")
                model = iResNet100()
                device = torch.device(f"cuda:{rank}")
                model = model.to(device)
                
                # Load checkpoint and remove module prefix if needed
                checkpoint = torch.load(model_path, map_location=device)
                checkpoint = remove_module_prefix(checkpoint)
                model.load_state_dict(checkpoint)

                # Save model state dict for sharing
                torch.save(model.state_dict(), temp_model_path)
            
            # Synchronize to ensure rank 0 saves the model first
            torch.distributed.barrier()

            # Load from saved file on other ranks
            if rank != 0:
                print(f"Rank {rank}: Waiting for temp file for {model_name}...")
                model = iResNet100()
                device = torch.device(f"cuda:{rank}")
                model = model.to(device)
                model.load_state_dict(torch.load(temp_model_path, map_location=device))

        # Wrap in DDP
        model = DDP(model, device_ids=[rank], output_device=rank)
        model.short_name = model_name.split("_")[0]  # Add short name for debugging
        models_to_test[model_name] = model

    return models_to_test


# CALFW: Positive and negative pairs based on age.
# CPLFW: Positive pairs with different poses.
# JEFF: Different facial expressions of the same subjects.
# CFP: Frontal and profile views of the same person.

# In[56]:


# Main worker function for each GPU
def main_worker(rank, world_size, model_filepaths, test_datasets, output_dir, subset_ratios, batch_size):
    try:
        print(f"Main worker started with rank: {rank}")
        # Initialize the process group for distributed training
        #dist.init_process_group("nccl", rank=rank, world_size=world_size)
        dist.init_process_group(
        backend="nccl", 
        init_method="env://", 
        #init_method="tcp://127.0.0.1:29500",
        rank=rank, 
        world_size=world_size, 
        #timeout=timedelta(seconds=1200)
        #timeout=timedelta(hours=48)
        timeout=timedelta(days=30)  # Adjust to your maximum expected runtime
        )

        if rank == 0:
            print(f"Starting distributed processing with {world_size} ranks.")

        dist.barrier()  # Synchronize all processes
        
        # Set the device based on rank
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        
        # Initialize models on the current device
        models_to_test = initialize_models(model_filepaths, rank, world_size)
        
        # Create DataLoaders with DistributedSampler for each dataset
        dataloaders = {
            dataset_name: DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True),
                num_workers=numworkers,
                drop_last=True
            )
            for dataset_name, dataset in test_datasets.items()
        }

        # Add the print statement here to check the sampled indices
        for dataset_name, dataset in test_datasets.items():
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, seed=42, drop_last=True)
            print(f"Rank {rank} is sampling indices: {list(sampler)}")
        
        # Run tests on the models and datasets
        run_tests(models_to_test, dataloaders, output_dir, subset_ratios, batch_size=batch_size,rank=rank)
    

    finally:
        # Clean up resources
        #for semaphore in mp.Semaphore._semaphore_tracker._semaphores.values():
            #semaphore.release()
                # Clean up the distributed environment
        torch.cuda.empty_cache()
        dist.destroy_process_group()
        #pass


# In[57]:


if __name__ == "__main__":
    # Define the environment variables needed for the distributed setup
    os.environ["MASTER_ADDR"] = "127.0.0.1"  # Use localhost for single-machine setup
    os.environ["MASTER_PORT"] = "29500"  # Use any free port, 29500 is common for PyTorch

    print(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR')}")
    print(f"MASTER_PORT: {os.environ.get('MASTER_PORT')}")
    os.environ["NCCL_DEBUG"] = "INFO"

  
    world_size = torch.cuda.device_count()  # Total number of GPUs

    model_filepaths = {
        model_name: os.path.join(model_save_path, subdirectory, filename)
        for model_name, filename in zip(model_names, model_filenames)
    }

    test_datasets = {
    'CPLFW': FlatDirectoryImageDataset(os.path.join(cplfw_extract_path, 'aligned images'), transform),
    'CALFW': FlatDirectoryImageDataset(os.path.join(calfw_extract_path, 'aligned images'), transform),
    'JEFF': FlatDirectoryImageDataset(jaffedbase_extract_path, transform),
    'CFP': ImageFolder(os.path.join(cfp_extract_path, 'Data/Images'), transform),
    }

    subset_ratios = {
    "CPF": subsetratio,    # % for CPF
    "CALFW": subsetratio,  # % for CALFW
    "JEFF": 1,  # % for JEFF
    "CPLFW": subsetratio  # % for CPLFW
    }

    # Spawn processes for each GPU, passing `rank` to each worker
    mp.spawn(main_worker, args=(world_size, model_filepaths, test_datasets, output_dir, subset_ratios, batchsize), nprocs=world_size, join=True)


# In[ ]:




