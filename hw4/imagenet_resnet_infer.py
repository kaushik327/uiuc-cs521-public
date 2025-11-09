import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from PIL import Image
import json
import os
import numpy as np

import skimage
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, spearmanr

from lime import lime
from smoothgrad import smoothgrad

# Load the pre-trained ResNet18 model
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.eval()  # Set model to evaluation mode

# Define the image preprocessing transformations
preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Load the ImageNet class index mapping
with open("imagenet_class_index.json") as f:
    class_idx = json.load(f)
idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
idx2synset = [class_idx[str(k)][0] for k in range(len(class_idx))]
id2label = {v[0]: v[1] for v in class_idx.values()}

imagenet_path = "./imagenet_samples"

# List of image file paths
image_paths = os.listdir(imagenet_path)

# Create output directory if it doesn't exist
output_dir = "./explanations"
os.makedirs(output_dir, exist_ok=True)


for img_path in image_paths:
    # Open and preprocess the image
    my_img = os.path.join(imagenet_path, img_path)
    input_image = Image.open(my_img).convert("RGB")
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(
        0
    )  # Create a mini-batch as expected by the model

    # Move the input and model to GPU if available
    if torch.cuda.is_available():
        input_batch = input_batch.to("cuda")
        model.to("cuda")

    # Perform inference
    with torch.no_grad():
        output = model(input_batch)

    # Get the predicted class index
    _, predicted_idx = torch.max(output, 1)
    predicted_idx = predicted_idx.item()
    predicted_synset = idx2synset[predicted_idx]
    predicted_label = idx2label[predicted_idx]

    print(f"Predicted label ({img_path}): {predicted_synset} ({predicted_label})")

    # LIME
    lime_mask = lime(
        input_tensor,
        model,
        predicted_idx,
        num_samples=1000,
        num_features=100,
        positive_only=True,
    )

    # SmoothGrad
    smoothgrad_mask = smoothgrad(input_tensor, model, num_samples=25)

    # Similarity
    print(f"{lime_mask.shape = }, {smoothgrad_mask.shape = }")

    lime_flat = lime_mask.flatten()
    smoothgrad_flat = smoothgrad_mask.flatten()

    kendall_tau, kendall_p = kendalltau(lime_flat, smoothgrad_flat)
    spearman_rho, spearman_p = spearmanr(lime_flat, smoothgrad_flat)

    # Create a figure with original image and LIME explanation
    fig, axes = plt.subplots(1, 4, figsize=(12, 5))

    input_tensor_for_display = (input_tensor - input_tensor.min()) / (
        input_tensor.max() - input_tensor.min()
    )
    input_tensor_for_display = input_tensor_for_display.permute(1, 2, 0)

    axes[0].imshow(input_image)
    axes[0].set_title(f"{img_path}\nPredicted {predicted_label}")
    axes[0].axis("off")

    axes[1].imshow(input_tensor_for_display)
    axes[1].set_title("Preprocessed input, scaled for display")
    axes[1].axis("off")

    axes[2].imshow(
        skimage.segmentation.mark_boundaries(
            input_tensor_for_display.numpy(), lime_mask
        )
    )
    axes[2].set_title(f"LIME Explanation boundaries")
    axes[2].axis("off")

    axes[3].imshow(smoothgrad_mask, cmap="gray")
    axes[3].set_title("SmoothGrad")
    axes[3].axis("off")

    plt.figtext(0.5, 0.2, f'''Similarity between explanations:
Kendall-Tau: {kendall_tau:.4f} (p={kendall_p:.4e})
Spearman: {spearman_rho:.4f} (p={spearman_p:.4e})''', ha='center', va='top')

    plt.tight_layout()

    # Save the figure
    output_filename = os.path.join(
        output_dir, f"expl_{os.path.splitext(img_path)[0]}.png"
    )
    plt.savefig(output_filename, bbox_inches="tight")
    plt.close()

    print(f"Saved explanations to: {output_filename}\n")
