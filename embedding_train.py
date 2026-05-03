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
    transforms.RandomResizedCrop(128, scale=(0.75, 1.0)),
    transforms.ColorJitter(0.2, 0.2, 0.2),
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
    def __init__(self, embedding_dim=128):
        super().__init__()

        base = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT
        )

        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.embedding = nn.Sequential(
            nn.Linear(576, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.embedding(x)

        # normalize for cosine similarity
        x = nn.functional.normalize(x, p=2, dim=1)
        return x


dataset = TripletItemDataset("datasets")
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = EmbeddingNet().to(device)

loss_fn = nn.TripletMarginLoss(margin=0.4)
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

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, loss={total_loss:.4f}")

torch.save(model.state_dict(), "item_embedding_model.pth")