import torch
from torch.nn import functional as F


def nt_xent_loss(z, temperature):
    """
    z: Tensor of shape (2, batch_size, latent_dim) containing the projections from the encoder
    temperature: A float scalar for the temperature parameter (larger values lead to a softer probability distribution)

    Computes the NT-Xent loss given the projection pairs z.
    NT-Xent: Normalized Temperature-scaled Cross-Entropy loss.
    SimCLR paper: https://arxiv.org/abs/2002.05709

    Returns:
    loss: A float scalar representing the NT-Xent loss
    """
    assert len(z.shape) == 3, "Input tensor must be 3D: (2, batch_size, latent_dim)"

    n_views, batch_sz, latent_dim = z.shape

    assert n_views == 2, "Input tensor must have two views for contrastive learning"

    # The cosine similarity matrix with shape (2N, 2N)
    # Each entry (i, j) represents the cosine similarity between z_i and z_j
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
