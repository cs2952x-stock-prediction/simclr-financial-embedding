import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    def __init__(
        self, sequences, labels, segment_length, step=1, device=torch.device("cpu")
    ):
        """
        Initializes the TimeSeriesDataset object.
        Should take a list of 2D data (NumPy arrays, torch tensors, or lists) for each time series.
        Each time series can have a different length.

        Parameters:
        - sequences (list of arrays/tensors/lists): List where each entry is a tensor with shape (seq_len, n_features).
        - labels (list of arrays/tensors/lists): List of labels for each time series.
        - segment_length (int): The length of each segment to sample.
        - step (int): The step size to use between consecutive segment samples.
        - device (torch.device): The device to use for the data.
        """
        self.sequences = []
        for seq in sequences:
            if isinstance(seq, torch.Tensor):
                self.sequences.append(seq.to(device))
            else:
                self.sequences.append(torch.tensor(seq, dtype=torch.float32).to(device))

        self.labels = []
        for label in labels:
            if isinstance(label, torch.Tensor):
                self.labels.append(label.to(device))
            else:
                self.labels.append(torch.tensor(label, dtype=torch.float32).to(device))

        self.segment_length = segment_length
        self.step = segment_length if step is None else step

        segment_counts = []
        for seq in self.sequences:
            n_segments = (len(seq) - segment_length) // self.step + 1
            segment_counts.append(n_segments)

        self.cum_lengths = torch.cumsum(torch.tensor(segment_counts), 0).to(device)

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

        sequence = self.sequences[seq_idx]
        labels = self.labels[seq_idx]

        start_idx = idx * self.step
        end_idx = start_idx + self.segment_length
        segment = sequence[start_idx:end_idx]
        labels = labels[start_idx:end_idx]

        # Splitting features and target
        return segment, labels
