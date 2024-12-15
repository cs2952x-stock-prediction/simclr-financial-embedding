import torch


def mask_with_added_gaussian(
    data,
    mask_prob=0.1,
    stat_axis=(1,),
    mask_axis=(0, 1),
    std_multiplier=0.01,
    means=None,
    stds=None,
):
    """
    Adds Gaussian noise to masked features in the data.

    Parameters:
    - data (tensor): Input array with shape (batch_sz, seq_len, n_features) or similar.
    - mask_prob (float): Probability of masking each element or example in the sequence. Default is 0.1.
    - stat_axis (int or tuple): Dimension(s) over which to calculate mean and std.
        By default, this is (1,) which corresponds to the sequence axis in data with shape (batch_sz, seq_len, n_features).
    - mask_axis (int or tuple): Dimension(s) along which to apply the mask.
        By default, this is (0, 1) which corresponds to the batch and sequence dimensions.
    - std_multiplier (float): Scaling factor for the standard deviation if calculated dynamically. Default is 0.01.
    - means (float or tensor, optional): The mean/bias of the added noise. Set to zero if None (equally likely to add/subtract from data).
    - stds (float or tensor, optional): Standard deviation of the added noise. Uses the standard deviation of the data if None.

    Returns:
    - ndarray: The input data with Gaussian noise added to selected elements or examples.
    """
    # Calculate means and stds across specified stat_axis if not provided
    if means is None:
        means = 0

    if stds is None:
        stds = torch.std(data, dim=stat_axis, keepdim=True) * std_multiplier

    # Generate Gaussian noise based on the calculated or provided means and stds
    noise = torch.randn_like(data) * stds + means

    # Generate mask based on mask_shape to apply masking by example, feature, or otherwise
    mask_shape = [
        data.shape[dim] if dim in mask_axis else 1 for dim in range(data.dim())
    ]
    mask = torch.rand(mask_shape) < mask_prob

    # Apply the noise only where the mask is True
    return torch.where(mask, data + noise, data).to(data.device)


def mask_with_multiplied_lognormal(
    data,
    mask_prob=0.1,
    stat_axis=(1,),
    mask_axis=(0, 1),
    std_multiplier=0.01,
    means=None,
    stds=None,
):
    """
    Applied lognormal noise by taking the product with masked features in the data.

    Parameters:
    - data (ndarray): Input array with shape (batch_sz, seq_len, n_features) or similar.
    - mask_prob (float): Probability of masking each element or example in the sequence. Default is 0.1.
    - stat_axis (int or tuple): Dimension(s) over which to calculate mean and std.
        By default, this is (1,) which corresponds to the sequence axis in data with shape (batch_sz, seq_len, n_features).
    - mask_axis (int or tuple): Dimension(s) along which to apply the mask.
        By default, this is (0, 1) which corresponds to the batch and sequence dimensions.
    - std_multiplier (float): Scaling factor for the standard deviation if calculated dynamically. Default is 0.01.
    - means (float or ndarray, optional): The mean/bias of the added noise. Set to zero if None (equally likely to add/subtract from data).
    - stds (float or ndarray, optional): Standard deviation of the added noise. Uses the standard deviation of the data if None.

    Returns:
    - ndarray: The input data with Lognormal noise applied to selected elements or examples.
    """
    # Calculate means and stds across specified stat_axis if not provided
    log_data = torch.log(data + 1e-8)  # Avoid log(0) by adding a small value
    if means is None:
        means = 0

    if stds is None:
        stds = torch.std(log_data, dim=stat_axis, keepdim=True) * std_multiplier

    # Generate lognormal noise based on the calculated or provided means and stds
    noise = torch.exp(torch.randn_like(*data.shape) * stds + means)

    # Generate mask based on mask_shape to apply masking by example, feature, or otherwise
    mask_shape = [
        data.shape[dim] if dim in mask_axis else 1 for dim in range(data.ndim)
    ]
    mask = torch.rand(*mask_shape) < mask_prob

    # Apply the noise only where the mask is True
    return torch.where(mask, data * noise, data)


def mask_with_constant(
    data,
    mask_prob=0.1,
    mask_axis=(0, 1),
    constant=0,
):
    """
    Replaces masked features with a contant value (or values if an array is provided)

    Parameters:
    - data (ndarray): Input array with shape (batch_sz, seq_len, n_features) or similar.
    - mask_prob (float): Probability of masking each element or example in the sequence. Default is 0.1.
    - mask_axis (int or tuple): Dimension(s) along which to apply the mask.
        By default, this is (0, 1) which corresponds to the batch and sequence dimensions.

    Returns:
    - ndarray: The input data with masked elements or examples replaced by a constant value.
    """
    # Generate mask based on mask_shape to apply masking by example, feature, or otherwise
    mask_shape = [
        data.shape[dim] if dim in mask_axis else 1 for dim in range(data.ndim)
    ]
    mask = torch.rand(*mask_shape) < mask_prob

    # Replaces with constant only where the mask is True
    return torch.where(mask, constant, data)


def mask_with_gaussian(
    data,
    mask_prob=0.1,
    stat_axis=(1,),
    mask_axis=(0, 1),
    std_multiplier=0.01,
    means=None,
    stds=None,
):
    """
    Replaces masked features with a value drawn from a Gaussian distribution.

    Parameters:
    - data (ndarray): Input array with shape (batch_sz, seq_len, n_features) or similar.
    - mask_prob (float): Probability of masking each element or example in the sequence. Default is 0.1.
    - stat_axis (int or tuple): Dimension(s) over which to calculate mean and std.
        By default, this is (1,) which corresponds to the sequence axis in data with shape (batch_sz, seq_len, n_features).
    - mask_axis (int or tuple): Dimension(s) along which to apply the mask.
        By default, this is (0, 1) which corresponds to the batch and sequence dimensions.
    - std_multiplier (float): Scaling factor for the standard deviation if calculated dynamically. Default is 0.01.
    - means (float or ndarray, optional): The mean/bias of the generated values. Uses the mean of the data if None.
    - stds (float or ndarray, optional): Standard deviation of the generated values. Uses the standard deviation of the data if None.

    Returns:
    - ndarray: The input data with Gaussian values replacing selected elements or examples.
    """
    # Calculate means and stds across specified stat_axis if not provided
    if means is None:
        means = torch.mean(data, dim=stat_axis, keepdim=True)

    if stds is None:
        stds = torch.std(data, dim=stat_axis, keepdim=True) * std_multiplier

    # Generate Gaussian noise based on the calculated or provided means and stds
    rand_vals = torch.randn_like(data) * stds + means

    # Generate mask based on mask_shape to apply masking by example, feature, or otherwise
    mask_shape = [
        data.shape[dim] if dim in mask_axis else 1 for dim in range(data.ndim)
    ]
    mask = torch.rand(*mask_shape) < mask_prob

    # Replace with random value only where the mask is True
    return torch.where(mask, rand_vals, data)  # data + noise, data)


def mask_with_lognormal(
    data,
    mask_prob=0.1,
    stat_axis=(1,),
    mask_axis=(0, 1),
    std_multiplier=0.01,
    means=None,
    stds=None,
):
    """
    Applied lognormal noise by taking the product with masked features in the data.

    Parameters:
    - data (ndarray): Input array with shape (batch_sz, seq_len, n_features) or similar.
    - mask_prob (float): Probability of masking each element or example in the sequence. Default is 0.1.
    - stat_axis (int or tuple): Dimension(s) over which to calculate mean and std.
        By default, this is (1,) which corresponds to the sequence axis in data with shape (batch_sz, seq_len, n_features).
    - mask_axis (int or tuple): Dimension(s) along which to apply the mask.
        By default, this is (0, 1) which corresponds to the batch and sequence dimensions.
    - std_multiplier (float): Scaling factor for the standard deviation if calculated dynamically. Default is 0.01.
    - means (float or ndarray, optional): The mean/bias of the generated values. Uses the mean of the data if None.
    - stds (float or ndarray, optional): Standard deviation of the generated values. Uses the standard deviation of the data if None.

    Returns:
    - ndarray: The input data with Lognormal values replacing selected elements or examples.
    """
    # Calculate means and stds across specified stat_axis if not provided
    log_data = torch.log(data + 1e-8)  # Avoid log(0) by adding a small value
    if means is None:
        means = torch.mean(log_data, dim=stat_axis, keepdim=True)

    if stds is None:
        stds = torch.std(log_data, dim=stat_axis, keepdim=True) * std_multiplier

    # Generate lognormal values based on the calculated or provided means and stds
    rand_vals = torch.exp(torch.randn_like(data) * stds + means)

    # Generate mask based on mask_shape to apply masking by example, feature, or otherwise
    mask_shape = [
        data.shape[dim] if dim in mask_axis else 1 for dim in range(data.ndim)
    ]
    mask = torch.rand(*mask_shape) < mask_prob

    # Replace with random values only where the mask is True
    return torch.where(mask, rand_vals, data)


def pairwise_swaps(data, swap_prob=0.1, axis=1):
    """
    Randomly swaps adjacent pairs of elements along the specified axis.

    Parameters:
    - data (ndarray): Input array with shape (batch_sz, seq_len, n_features) or similar.
    - swap_prob (float): Probability of shuffling each pair of adjacent elements. Default is 0.1.
    - axis (int): Dimension along which to shuffle pairs of elements.

    Returns:
    - ndarray: The input data with randomm pairs of adjacent elements swapped along the specified axis.
    """
    # Clone to avoid modifying the original in-place
    data_swapped = data.clone()

    # Step 1: Move the specified axis -> dimension=1 using permute().
    # This ensures the axis we want to swap is dimension 1, making the code simpler.
    ndims = data_swapped.ndim
    dims = list(range(ndims))
    # Swap 'axis' with '1' in dims
    dims[axis], dims[1] = dims[1], dims[axis]
    data_reoriented = data_swapped.permute(
        *dims
    )  # Now shape is: (..., L, ...), with L at dim=1

    # L is the length along the (now) dimension=1
    L = data_reoriented.shape[1]

    # Step 2: Iterate over adjacent pairs along dimension=1
    for i in range(L - 1):
        # Step 3: Make a random boolean mask (one per "batch" in dim=0),
        # deciding whether to swap this pair (i, i+1).
        # If data is 3D, dim=0 is typically the batch dimension.
        mask = torch.rand(data_reoriented.shape[0], device=data.device) < swap_prob

        # Step 4: Swap the entire "row" (covering all features) if mask[b] == True
        tmp = data_reoriented[mask, i, :].clone()
        data_reoriented[mask, i, :] = data_reoriented[mask, i + 1, :]
        data_reoriented[mask, i + 1, :] = tmp

    # Step 5: Move the axis back to its original position
    dims[axis], dims[1] = dims[1], dims[axis]
    data_swapped = data_reoriented.permute(*dims)

    return data_swapped


def mask_with_smoothing(data, smooth_prob=0.1, window_sz=3, axis=1):
    """
    Smooths the data by averaging adjacent pairs of elements along the specified axis.

    Parameters:
    - data (ndarray): Input array with shape (batch_sz, seq_len, n_features) or similar.
    - smooth_prob (float): Probability of smoothing each pair of adjacent elements. Default is 0.1.
    - window_sz (int): Size of the window for smoothing. Default is 3.
    - axis (int): Dimension along which to smooth pairs of elements.

    Returns:
    - ndarray: The input data with random pairs of adjacent elements smoothed along the specified axis.
    """
    # Clone to avoid modifying the original in-place
    data_smoothed = data.clone()

    # Step 1: Move the specified axis -> dimension=1 using permute().
    # This ensures the axis we want to smooth is dimension 1, making the code simpler.
    ndims = data_smoothed.ndim
    dims = list(range(ndims))
    # Swap 'axis' with '1' in dims
    dims[axis], dims[1] = dims[1], dims[axis]
    data_reoriented = data_smoothed.permute(
        *dims
    )  # Now shape is: (..., L, ...), with L at dim=1

    # L is the length along the (now) dimension=1
    L = data_reoriented.shape[1]

    # Step 2: Iterate over adjacent pairs along dimension=1
    for i in range(L - 1):
        # Step 3: Make a random boolean mask (one per "batch" in dim=0),
        # deciding whether to smooth this pair (i, i+1).
        # If data is 3D, dim=0 is typically the batch dimension.
        mask = torch.rand(data_reoriented.shape[0], device=data.device) < smooth_prob

        # Step 4: Smooth the entire "row" (covering all features) if mask[b] == True
        data_reoriented[mask, i, :] = torch.mean(
            data_reoriented[mask, i : i + window_sz, :], dim=1, keepdim=False
        )

    # Step 5: Move the axis back to its original position
    dims[axis], dims[1] = dims[1], dims[axis]
    data_smoothed = data_reoriented.permute(*dims)

    return data_smoothed
