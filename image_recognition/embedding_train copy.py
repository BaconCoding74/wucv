import os

import torch
import ir_constants as irc
import torch.nn as nn

from model_class import EmbeddingNet, TripletItemDataset
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("WARNING: CUDA not available, running on CPU!")
    print(f"  torch version: {torch.__version__}")
    print(f"  torch CUDA version: {torch.version.cuda}")
else:
    print(f"Running on GPU: {torch.cuda.get_device_name(0)}")

# Batch size is 32 because provide average of multiple samples instead of single sample
if __name__ == "__main__":
    dataset = TripletItemDataset(f"{irc.DATASETS_PATH}/{irc.SELECTED_DATASET}")

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,        # parallel CPU workers for loading
        pin_memory=True,      # faster CPU→GPU transfer
        prefetch_factor=2,    # load next batch while GPU is busy
        persistent_workers=True  # don't restart workers each epoch
    )

    model = EmbeddingNet().to(device)

    # Margin define minimum distance between anchor-positve and anchor-negative pairs, 
    # else model will be penalized due to loss > 0
    loss_fn = nn.TripletMarginLoss(margin=0.4)

    # Define what optimizer to use and learning rate
    # Learning rate is step size when updating weights which affecting the speed of training
    # The algorithm is just different in computing new weights
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(1):
        model.train()
        total_loss = 0

        for anchor, positive, negative in loader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            
            print("Batch loaded:")
            print("\tanchor.shape:", anchor.shape)
            print("\tpositive.shape:", positive.shape)
            print("\tnegative.shape:", negative.shape)

