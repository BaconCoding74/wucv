import random
import torch.nn as nn
import ir_constants as irc
from PIL import Image
from pathlib import Path
from torchvision import transforms, models
from torch.utils.data import Dataset

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
class HardTripletItemDataset(TripletItemDataset):
    def __init__(self, root):
        super().__init__(root)

        self.similar_items = {
            "flower_petal": 3,
            "pentagon": 4,
            "star": 5,
            "yellow_star": 3,
            "yoyo": 3,
            "mask": 3,
        }

        self.similar_labels = {}
        self.other_labels = {}

        for label in self.similar_items.keys():
            if label not in self.labels:
                raise ValueError(f"Label '{label}' specified in similar_items not found in dataset.")


            self.similar_labels[label] = [
                l for l in self.labels if l.startswith(label) and l != label
            ]

            self.other_labels[label] = [
                l for l in self.labels if not l.startswith(label) and l != label
            ]

    def __getitem__(self, idx):
        anchor_label = random.choice(self.labels)

        positive_candidates = self.images[anchor_label]
        positive_paths = random.sample(positive_candidates, irc.MODEL_NUM_P_IMAGE + 1)
        anchor_path, positive_paths = positive_paths[0], positive_paths[1:]

        base_label = anchor_label.rsplit('_', 1)[0]

        similar_negative_candidates = self.similar_labels.get(base_label, [])
        other_negative_candidates = self.other_labels.get(base_label, [])
        
        negative_paths = []
        for i in range(irc.MODEL_NUM_N_IMAGE):
            if len(similar_negative_candidates) > 0 and random.random() < 0.7:
                negative_label = random.choice(similar_negative_candidates)
                similar_negative_candidates.remove(negative_label)

            else:
                negative_label = random.choice(other_negative_candidates)
                other_negative_candidates.remove(negative_label)
            
            negative_paths.append(random.choice(self.images[negative_label]))

        anchor = tf(Image.open(anchor_path).convert("RGB"))
        positives = [tf(Image.open(p).convert("RGB")) for p in positive_paths]
        negatives = [tf(Image.open(n).convert("RGB")) for n in negative_paths]

        return anchor, positives, negatives

        
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
    
class HardTripletLossEmbeddingNet(EmbeddingNet):
    def __init__(self, embedding_dim=128):
        super().__init__(embedding_dim)

    def compute_loss(self, device, anchor, positives, negatives):
        anchor_emb = super().forward(anchor.to(device))
        positive_embs = [super().forward(p.to(device)) for p in positives]
        negative_embs = [super().forward(n.to(device)) for n in negatives]

        return anchor_emb, positive_embs, negative_embs