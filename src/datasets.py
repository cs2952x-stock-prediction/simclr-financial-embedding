import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, segment_length, step=1, device=torch.device("cpu")):
        """
        Initializes the TimeSeriesDataset object.
        Should take a list of 2D data (NumPy arrays, torch tensors, or lists) for each time series.
        Each time series can have a different length.

        Parameters:
        - sequences (list of arrays/tensors/lists): List where each entry is a tensor with shape (seq_len, n_features).
        - segment_length (int): The length of each segment to sample.
        - step (int): The step size to use between consecutive segment samples.
        - device (torch.device): The device to use for the data.
        """
        self.sequences = [
            (
                torch.tensor(seq, dtype=torch.float32).to(device)
                if not isinstance(seq, torch.Tensor)
                else seq.to(device)
            )
            for seq in sequences
        ]
        self.segment_length = segment_length
        self.step = segment_length if step is None else step
        segment_counts = [
            (len(seq) - self.segment_length) // self.step + 1 for seq in self.sequences
        ]
        self.cum_lengths = torch.cumsum(
            torch.tensor(
                segment_counts,
            ),
            0,
        )

    def __len__(self):
        """
        Returns the total number of segments that can be sampled from the dataset.
        """
        return int(self.cum_lengths[-1].item())

    def __getitem__(self, idx):
        """
        Returns the segment at the given index.
        """
        seq_idx = torch.searchsorted(
            self.cum_lengths, torch.tensor(idx), right=True
        ).item()
        seq_idx = int(seq_idx)

        if seq_idx > 0:
            idx -= self.cum_lengths[seq_idx - 1].item()

        seq = self.sequences[seq_idx]
        start_idx = idx * self.step
        end_idx = start_idx + self.segment_length
        segment = seq[start_idx:end_idx]

        # Splitting features and target
        return segment[:, :-1], segment[:, -1]
