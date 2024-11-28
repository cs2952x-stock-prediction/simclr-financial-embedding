import os
import random

import pandas as pd
import torch
from torch.utils.data import Dataset

F32 = torch.float32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, labels, segment_length, step=1):
        """
        Initializes the TimeSeriesDataset object.
        Should take a list of 2D data (NumPy arrays, torch tensors, or lists) for each time series.
        Each time series can have a different length.

        Parameters:
        - sequences (list of arrays/tensors/lists): List where each entry is a tensor with shape (seq_len, n_features).
        - labels (list of arrays/tensors/lists): List of labels for each time series.
        - segment_length (int): The length of each segment to sample.
        - step (int): The step size to use between consecutive segment samples.
        """
        self.sequences = []
        for seq in sequences:
            if isinstance(seq, torch.Tensor):
                self.sequences.append(seq.to(DEVICE))
            else:
                self.sequences.append(torch.tensor(seq, dtype=F32).to(DEVICE))

        self.labels = []
        for label in labels:
            if isinstance(label, torch.Tensor):
                self.labels.append(label.to(DEVICE))
            else:
                self.labels.append(torch.tensor(label, dtype=F32).to(DEVICE))

        self.segment_length = segment_length
        self.step = segment_length if step is None else step

        segment_counts = []
        for seq in self.sequences:
            n_segments = (len(seq) - segment_length) // self.step + 1
            segment_counts.append(n_segments)

        self.cum_lengths = torch.cumsum(torch.tensor(segment_counts), 0).to(DEVICE)

    def __len__(self):
        """
        Returns the total number of segments that can be sampled from the dataset.
        """
        return int(self.cum_lengths[-1].item())

    def __getitem__(self, idx):
        """
        Returns the segment at the given index.
        """
        seq_idx = torch.searchsorted(self.cum_lengths, torch.tensor(idx), right=True)
        seq_idx = int(seq_idx.item())

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


class BigSeriesDataset(Dataset):
    def __init__(
        self,
        files,
        label,
        segment_length,
        step=1,
        shuffle=True,
    ):
        """
        Initializes the BigSeriesDataset object.
        Should take a list of file paths to load the data from.
        Loads a single CSV file at a time and samples segments from it.
        This allows training on datasets that are too large to fit in memory.

        Parameters:
        - files (list or str): List of file paths to load the data from or a directory containing all files.
        - label (str): The column name of the label in the CSV files
        - segment_length (int): The length of each segment to sample.
        - step (int): The step size to use between consecutive segment samples.
        - shuffle (bool): Whether to shuffle the data.
        """
        self.files = files
        self.label = label
        self.segment_length = segment_length
        self.step = step
        self.shuffle = shuffle

        # If a directory is provided, load all CSV files in the directory
        if type(files) == str:
            files = [
                os.path.join(files, f) for f in os.listdir(files) if f.endswith(".csv")
            ]

        #
        self.file_segments = {}
        for file_idx, file_path in enumerate(files):
            length = len(pd.read_csv(file_path))
            n_segments = (length - segment_length) // step + 1

            if n_segments <= 1:
                raise ValueError(
                    f"File {file_path} is too short to sample segments of length {segment_length}"
                )

            self.file_segments[file_idx] = [
                (i * step, i * step + segment_length) for i in range(n_segments)
            ]

        self.current_file_idx = 0
        self.current_file = self.files[0]

        df = pd.read_csv(self.current_file)
        x_cols = [col for col in df.columns if col != self.label]
        y_cols = [self.label]

        self.current_data = torch.tensor(df[x_cols].values, dtype=F32).to(DEVICE)
        self.current_labels = torch.tensor(df[y_cols].values, dtype=F32).to(DEVICE)

        if self.shuffle:
            random.shuffle(self.file_segments[0])

        self.cum_lengths = torch.cumsum(
            torch.tensor([len(segments) for segments in self.file_segments.values()]),
            0,
        ).to(DEVICE)

    def __len__(self):
        """
        Returns the total number of segments that can be sampled from the dataset.
        """
        return int(self.cum_lengths[-1].item())

    def __getitem__(self, idx):
        """
        Returns the segment at the given index.
        If shuffle is enabled, the segments will be shuffled within each file.

        Parameters:
        - idx (int): The index of the segment to retrieve.
                     Note that this is the index of the segment, not the index of the data point.
                     Additionally, if shuffle is enabled, the segments for each file will be out of order.
        """
        file_idx = torch.searchsorted(self.cum_lengths, torch.tensor(idx), right=True)
        file_idx = int(file_idx.item())

        if file_idx > 0:
            idx -= self.cum_lengths[file_idx - 1].item()

        if file_idx != self.current_file_idx:
            self.current_file_idx = file_idx
            self.current_file = self.files[file_idx]

            df = pd.read_csv(self.current_file)
            x_cols = [col for col in df.columns if col != self.label]
            y_cols = [self.label]

            self.current_data = torch.tensor(df[x_cols].values, dtype=F32).to(DEVICE)
            self.current_labels = torch.tensor(df[y_cols].values, dtype=F32).to(DEVICE)

            if self.shuffle:
                random.shuffle(self.file_segments[file_idx])
            else:
                self.file_segments[file_idx].sort()

        start, end = self.file_segments[file_idx][idx]
        segment = self.current_data[start:end]
        labels = self.current_labels[start:end]

        return segment, labels
