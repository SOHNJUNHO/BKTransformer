import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


def validate_split_ratios(train_split, val_split, test_split):
    splits = {
        "train_split": train_split,
        "val_split": val_split,
        "test_split": test_split,
    }
    for name, value in splits.items():
        if not 0 < value < 1:
            raise ValueError(f"{name} must be between 0 and 1.")

    total = train_split + val_split + test_split
    if not np.isclose(total, 1.0):
        raise ValueError("train_split + val_split + test_split must equal 1.0.")


def create_sequences(df, block_size=818):
    sequences = []
    group_keys = []

    for user_id, group in df.groupby("user_id"):
        seq = group[["skill_id", "correct"]].values

        if len(seq) > block_size:
            seq = seq[:block_size]

        sequences.append(torch.tensor(seq, dtype=torch.float32))
        group_keys.append(user_id)

    return sequences, group_keys


class BKTSequenceDataset(Dataset):
    def __init__(self, sequences, group_keys):
        self.sequences = sequences
        self.group_keys = group_keys

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.group_keys[idx]


def bkt_collate_fn(batch):
    sequences = [item[0] for item in batch]
    keys = [item[1] for item in batch]

    padded = pad_sequence(sequences, batch_first=True, padding_value=-1000)

    obs = padded[:, :-1, :]
    output = padded[:, 1:, :]

    return obs, output, keys


def get_data_loaders(
    df,
    block_size=818,
    batch_size=32,
    train_split=0.8,
    val_split=0.1,
    test_split=0.1,
    seed=42,
):
    validate_split_ratios(train_split, val_split, test_split)

    sequences, group_keys = create_sequences(
        df,
        block_size=block_size,
    )

    np.random.seed(seed)
    indices = np.arange(len(sequences))
    np.random.shuffle(indices)

    n = len(indices)
    train_end = int(train_split * n)
    val_end = train_end + int(val_split * n)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    train_sequences = [sequences[i] for i in train_idx]
    val_sequences = [sequences[i] for i in val_idx]
    test_sequences = [sequences[i] for i in test_idx]

    train_group_keys = [group_keys[i] for i in train_idx]
    val_group_keys = [group_keys[i] for i in val_idx]
    test_group_keys = [group_keys[i] for i in test_idx]

    train_dataset = BKTSequenceDataset(train_sequences, train_group_keys)
    val_dataset = BKTSequenceDataset(val_sequences, val_group_keys)
    test_dataset = BKTSequenceDataset(test_sequences, test_group_keys)

    train_lengths = [len(seq) for seq in train_sequences]
    weights = torch.tensor([np.log(length + 1) for length in train_lengths], dtype=torch.float32)
    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(weights),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=bkt_collate_fn,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=bkt_collate_fn,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=bkt_collate_fn,
        num_workers=0,
    )

    return train_loader, val_loader, test_loader
