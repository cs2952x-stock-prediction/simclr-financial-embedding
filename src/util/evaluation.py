import torch
from torch.nn import functional as F


def nt_xent_loss(z_i, z_j, temperature):
    """
    z: Tensor of shape (2, batch_size, latent_dim) containing the projections from the encoder
    temperature: A float scalar for the temperature parameter (larger values lead to a softer probability distribution)

    Computes the NT-Xent loss given the projection pairs z.
    NT-Xent: Normalized Temperature-scaled Cross-Entropy loss.
    SimCLR paper: https://arxiv.org/abs/2002.05709

    Returns:
    loss: A float scalar representing the NT-Xent loss
    """
    assert (
        len(z_i.shape) == 2
    ), f"Tensors must have 2 axis (batch_sz, latent_dim), found shape {z_i.shape}"

    assert (
        z_i.shape == z_j.shape
    ), f"Tensors must have the same shape, found shapes ${z_i.shape} and {z_j.shape}"

    batch_sz, latent_dim = z_i.shape

    # The cosine similarity matrix with shape (2N, 2N)
    # Each entry (i, j) represents the cosine similarity between z_i and z_j
    z = torch.cat([z_i, z_j], dim=0)
    z = z.reshape(2 * batch_sz, latent_dim)
    pair_similarities = F.cosine_similarity(z[None, :, :], z[:, None, :], dim=-1)

    # This disregards the similarity at the same index (i.e. when i = j)
    eye = torch.eye(2 * batch_sz, device=z.device)
    pair_similarities[eye.bool()] = float("-inf")

    # The similarity between the views of the same image is used as a positive pair
    # view i and i + N are positive pairs if i < N
    # view i and i - N are positive pairs if i >= N
    target = torch.arange(2 * batch_sz, device=z.device)
    target[:batch_sz] += batch_sz
    target[batch_sz:] -= batch_sz

    return F.cross_entropy(pair_similarities / temperature, target, reduction="mean")


def average_percentage_error(true_log_diff, pred_log_diff):
    """
    Compute the Average Percentage Error (APE) in PyTorch.

    Args:
        true_log_diff (torch.Tensor): True log differences (log(p_next) - log(p_current)).
        pred_log_diff (torch.Tensor): Predicted log differences (log(p_pred) - log(p_current)).

    Returns:
        torch.Tensor: The Average Percentage Error (APE) as a scalar.
    """
    # Convert log differences to price ratios
    true_ratios = torch.exp(true_log_diff)
    pred_ratios = torch.exp(pred_log_diff)

    # Compute percentage error
    percentage_errors = torch.abs((true_ratios - pred_ratios) / true_ratios) * 100

    # Compute average percentage error
    return percentage_errors.mean()
