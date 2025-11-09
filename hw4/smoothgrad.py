import torch


def smoothgrad(
    image: torch.Tensor,
    model: torch.nn.Module,
    num_samples: int,
) -> torch.Tensor:
    std = 0.15 * (torch.max(image).item() - torch.min(image).item())
    total = torch.zeros_like(image, dtype=torch.float32)

    for _ in range(num_samples):
        # add gaussian noise
        noisy_image = image + torch.randn_like(image) * std
        x = (
            noisy_image.to(dtype=torch.float32)
            .unsqueeze(0)
            .detach()
            .requires_grad_(True)
        )

        # get gradients
        output = model(x)
        grad = torch.autograd.grad(output.sum(), x)[0].detach()

        # accumulate gradients by magnitude
        total += grad[0] ** 2

    grad_mask = total / num_samples
    grad_mask = grad_mask.permute(1, 2, 0)

    # flatten to 2d and scale for display
    grad_2d = torch.max(torch.abs(grad_mask), dim=2).values
    scaled = (grad_2d - grad_2d.min()) / (grad_2d.quantile(0.99) - grad_2d.min())
    return torch.clamp(scaled, 0, 1)
