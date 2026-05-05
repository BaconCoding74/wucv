import random
import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

tf = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.RandomResizedCrop(128, scale=(0.35, 1.0), ratio=(0.8, 1.25)),
    transforms.RandomApply([
        transforms.ColorJitter(
            # Just changing magnitude of RGB channels
            brightness=0.3,

            # Washing out or sharpen the edge
            contrast=0.3,
            
            # Averaging value of RGB channels 
            # gray = avg(R, G, B)
            # pixel_R = gray + saturation * (original_R - gray)
            # ...
            saturation=0.25, 

            # Shift hue a little bit which help when color change a little
            hue=0.05,
        ),
    ], p=0.8),
    transforms.RandomAffine(
        degrees=5,
        translate=(0.08, 0.08),
        scale=(0.9, 1.15),
    ),
    transforms.ToTensor(),
])

class TripletItemDataset(Dataset):
    def __init__(self, root):
        self.root = Path(root)
        self.classes = [p for p in self.root.iterdir() if p.is_dir()]
        self.images = {
            c.name: list(c.glob("*.*"))
            for c in self.classes
        }
        self.labels = list(self.images.keys())

    def __len__(self):
        return 5000

    def __getitem__(self, idx):
        anchor_label = random.choice(self.labels)

        positive_candidates = self.images[anchor_label]
        anchor_path, positive_path = random.sample(positive_candidates, 2)

        negative_label = random.choice([l for l in self.labels if l != anchor_label])
        negative_path = random.choice(self.images[negative_label])

        anchor = tf(Image.open(anchor_path).convert("RGB"))
        positive = tf(Image.open(positive_path).convert("RGB"))
        negative = tf(Image.open(negative_path).convert("RGB"))

        return anchor, positive, negative

class EmbeddingNet(nn.Module):
    # Embedding_dim defined the length of output vector
    # The reason not 1 is because what we need is features of image, not label
    # The reason why is 128 is because it is sufficient to capture features of images
    def __init__(self, embedding_dim=128):
        super().__init__()
        
        # Use pretrained mobilenet_v3_small because it works well in extracting features
        # even it does not know our game assets
        base = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT
        )

        # There is base.features and base.classifier
        # base.features provide visual feature map (What we need)
        # base.classifier provide label prediction
        # base.features produce (B, C, H, W) 
        # where B = batch size, C = channels, H = height, W = width
        # Imagine C is features of image, H and W are spatial dimensions for these images
        # Normally size of input image will be reduced through downsampling
        # For mobilenet_v3_small the C = 576
        self.features = base.features

        # Reduce spatial dimensions to 1x1, so output is (B, C, 1, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Move through layers
        # From ChatGPT it shows that (need verification)
        # The input of first layer is output channel of mobilenet_v3_small
        # The reason we need to train layer again because pretrain layer only extract features
        # Now we need layer to learn how to use these features
        self.embedding = nn.Sequential(
            # Combine extracted features and reassign the weights
            nn.Linear(576, 256),

            # RELU breaks linearity to allow model to learn complex patterns by introducing rules below
            # If input > 0, output = input
            # If input <= 0, output = 0
            # Output generated is based on the conditon
            nn.ReLU(),

            # Same as first layer
            nn.Linear(256, embedding_dim)
        )

    # Will be called automatically when we call model(input)
    def forward(self, x):
        # Extract features through mobilenet_v3_small
        x = self.features(x)

        # Flatten from (B, C, 1, 1) to (B, C)
        # Ex: [[0.1, 0.2, ..., 0.5]] to [0.1, 0.2, ..., 0.5]
        x = self.pool(x).flatten(1)

        # Passing defined layers
        x = self.embedding(x)

        # Normalize so the distance of anchor-positive and anchor-negative based on angle instead of both magnitude and angle
        x = nn.functional.normalize(x, p=2, dim=1)
        return x


dataset = TripletItemDataset("datasets")
# Batch size is 32 because provide average of multiple samples instead of single sample
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = EmbeddingNet().to(device)

# Margin define minimum distance between anchor-positve and anchor-negative pairs, 
# else model will be penalized due to loss > 0
loss_fn = nn.TripletMarginLoss(margin=0.4)

# Define what optimizer to use and learning rate
# Learning rate is step size when updating weights which affecting the speed of training
# The algorithm is just different in computing new weights
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(10):
    model.train()
    total_loss = 0

    for anchor, positive, negative in loader:
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        a = model(anchor)
        p = model(positive)
        n = model(negative)

        loss = loss_fn(a, p, n)

        # Clean up gradients
        optimizer.zero_grad()

        # Comptute gradients for all parameter gradient = (d(loss)) / (d(weights))
        loss.backward()

        # Update weights for each parameter
        # formula based on optimizer
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, loss={total_loss:.4f}")

torch.save(model.state_dict(), "item_recognition_3.pth")