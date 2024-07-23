import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from dataload import PairedDataset
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from tqdm import tqdm
import pdb

# Image data transformation
image_transform = transforms.Compose([
    transforms.Resize((368, 368)),
    transforms.ToTensor(),
])

# Load datasets
radar_range = [200, 600]
radar_normalize = False

# Paths to data
data_paths = {
    'train': ('/home/yunkwan/project/radarclip/data_train/image', '/home/yunkwan/project/radarclip/data_train/radar'),
    'val': ('/home/yunkwan/project/radarclip/data_val/image', '/home/yunkwan/project/radarclip/data_val/radar'),
    'test': ('/home/yunkwan/project/radarclip/data_test/image', '/home/yunkwan/project/radarclip/data_test/radar')
}

# Dataset and DataLoader setup
datasets = {key: PairedDataset(image_root_dir=paths[0], radar_root_dir=paths[1], radar_range=radar_range,
                               transform=image_transform, radar_normalize=radar_normalize) for key, paths in data_paths.items()}
loaders = {key: DataLoader(dataset, batch_size=256, shuffle=(key=='train')) for key, dataset in datasets.items()}

# Calculate similarity
def calculate_similarity(radar1, radar2, method='cosine'):
    radar1_flat = radar1.flatten()
    radar2_flat = radar2.flatten()

    if method == 'cosine':
        return np.dot(radar1_flat, radar2_flat) / (np.linalg.norm(radar1_flat) * np.linalg.norm(radar2_flat))
    elif method == 'mse':
        return np.mean((radar1_flat - radar2_flat) ** 2)
    elif method == 'dtw':
        radar1_flat = radar1_flat.reshape(1, -1)
        radar2_flat = radar2_flat.reshape(1, -1)
        distance, _ = fastdtw(radar1_flat, radar2_flat, dist=euclidean)
        return -distance

# Find target
def find_target(loader, target_label, target_index):
    count = 0
    target_radar, target_image = None, None
    with torch.no_grad():
        for images, radars, _, _, labels in tqdm(loader, desc="Searching for target"):
            for i, label in enumerate(labels):
                if label == target_label:
                    if count == target_index:
                        target_radar = radars[i]
                        target_image = images[i]
                        return target_image, target_radar
                    count += 1
    raise ValueError("Target index out of range.")

# Find most similar
def find_most_similar(loader, target_radar, compare_label, method='cosine'):
    best_similarity = -float('inf') if method == 'cosine' else float('inf')
    best_radar, best_image = None, None
    with torch.no_grad():
        for images, radars, _, _, labels in tqdm(loader, desc=f"Comparing with label {compare_label}"):
            for i, label in enumerate(labels):
                if label == compare_label:
                    similarity = calculate_similarity(target_radar, radars[i], method)
                    if (method == 'cosine' and similarity > best_similarity) or (method != 'cosine' and similarity < best_similarity):
                        best_similarity = similarity
                        best_radar = radars[i]
                        best_image = images[i]
    return best_image, best_radar, best_similarity

# Visualize results
def visualize_results(images, radars, similarities, titles, method, ylim=0.006):
    fig, axs = plt.subplots(2, len(images), figsize=(15, 10))
    for i, (image, radar) in enumerate(zip(images, radars)):
        axs[0, i].plot(radar.flatten(), color='blue', label=f"Label: {titles[i]} - Similarity: {similarities[i]:.4f}")
        axs[0, i].set_ylim(-ylim, ylim)
        axs[0, i].legend()
        axs[1, i].imshow(np.transpose(image.numpy(), (1, 2, 0)))
        axs[1, i].set_title(titles[i])
    plt.tight_layout()
    plt.show()
    plt.savefig(f'/home/yunkwan/project/radarclip/similar_plot_{method}')

# Process comparisons
def process_comparisons(loader, index_0=4500, method='cosine', ylim=0.006):
    target_image, target_radar = find_target(loader, 0, index_0)
    best_image_1, best_radar_1, sim_1 = find_most_similar(loader, target_radar, 1, method)
    best_image_2, best_radar_2, sim_2 = find_most_similar(loader, target_radar, 2, method)

    visualize_results(
        [target_image, best_image_1, best_image_2],
        [target_radar, best_radar_1, best_radar_2],
        [1.0, sim_1, sim_2],
        ["Target (Label 0)", "Most Similar (Label 1)", "Most Similar (Label 2)"],
        method, ylim
    )

# Example usage
process_comparisons(loaders['train'], index_0=300, method='cosine')
