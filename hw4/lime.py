import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import pairwise_distances
from skimage.segmentation import quickshift
import torch
from tqdm import tqdm


def lime(
    image: torch.Tensor,
    model: torch.nn.Module,
    target_label: int,
    num_samples: int = 1000,
    num_features: int = 5,
    positive_only=True,
):
    # superpixel segmentation
    # TODO: try other options from scikit-image
    image_np = image.permute(1, 2, 0).cpu().numpy()
    segments = quickshift(
        image_np, kernel_size=4, max_dist=200, ratio=0.2, convert2lab=False
    )
    n_features = np.unique(segments).shape[0]

    # random binary perturbations for superpixels, where the first is the original
    perturbations = np.random.randint(0, 2, (num_samples, n_features))
    perturbations[0, :] = 1

    labels = []
    for row in tqdm(perturbations, desc="Doing LIME perturbations"):
        # create perturbed image: start with original, black out turned-off superpixels
        perturbed = image.clone()
        for superpixel_id in np.where(row == 0)[0]:
            perturbed[:, segments == superpixel_id] = 0

        # get label
        with torch.no_grad():
            output = model(perturbed.unsqueeze(0))
            pred = output.detach().cpu().numpy()[0]
        labels.append(pred[target_label])

    labels = np.array(labels)

    # compute classifier weights based on distance of binary mask to all-ones
    distances = pairwise_distances(
        perturbations, np.ones((1, n_features)), metric="cosine"
    )
    weights = 1 - distances.flatten()

    # fit interpretable model
    model = Ridge(alpha=1, fit_intercept=True)
    model.fit(perturbations, labels, sample_weight=weights)

    # get top features sorted by absolute coefficient value
    feature_weights = sorted(
        zip(range(n_features), model.coef_), key=lambda x: np.abs(x[1]), reverse=True
    )[:num_features]

    mask = np.zeros(segments.shape, dtype=int)

    for feature_id, weight in feature_weights:
        if positive_only and weight <= 0:
            continue
        mask[segments == feature_id] = 1

    print(f"{np.sum(mask > 0)}/{mask.size} pixels selected")
    return mask
