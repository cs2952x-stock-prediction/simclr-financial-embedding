import numpy as np
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, series, segment_length, step=1, transform=None):
        """
        Initializes the TimeSeriesDataset object.
        Should take a list of 2D ndarrays, where each ndarray is a time series.
        Because series is a list of arrays, each time series can have a different length.

        Parameters:
        - series (list of ndarrays): List where each entry is an ndarray with shape (seq_len, n_features).
        - segment_length (int): The length of each segment to sample.
        - step (int): The step size to use between consecutive segment samples.
        - transform (callable, optional): Optional transform to apply to each segment.
        """
        self.series = series
        self.segment_length = segment_length
        self.step = segment_length if step is None else step
        self.cum_lengths = np.cumsum(
            [(len(seq) - self.segment_length) // self.step + 1 for seq in series]
        )  # Cumulative lengths of the series --- makes it easier to sample from the dataset
        self.transform = transform

    def __len__(self):
        """
        Returns the total number of segments that can be sampled from the dataset.
        This is not the same as the number of sequences in the dataset or the number of datum over the entire dataset.
        """
        return self.cum_lengths[-1]

    def __getitem__(self, idx):
        """
        Returns the segment at the given index.
        """
        seq_idx = np.searchsorted(self.cum_lengths, idx, side="right")
        if seq_idx > 0:
            idx -= self.cum_lengths[seq_idx - 1]
        seq = self.series[seq_idx]
        start_idx = idx * self.step
        end_idx = start_idx + self.segment_length
        segment = seq[start_idx:end_idx]
        if self.transform is not None:
            segment = self.transform(segment)
        return segment
