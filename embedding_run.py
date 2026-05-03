import random
import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from time import perf_counter

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

model = EmbeddingNet().to(device)
model.load_state_dict(torch.load("item_embedding_model.pth", map_location=device))
model.eval()

test_tf = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

def encode(path):
    img = test_tf(Image.open(path).convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model(img)

    return emb


# build database
reference_db = {}

for item_folder in Path("reference_items").iterdir():
    if not item_folder.is_dir():
        continue

    embeddings = []

    for img_path in item_folder.glob("*.*"):
        embeddings.append(encode(img_path))

    reference_db[item_folder.name] = torch.mean(torch.cat(embeddings), dim=0, keepdim=True)


def find_item(query_path):
    query = encode(query_path)

    scores = []

    for name, ref_emb in reference_db.items():
        score = torch.cosine_similarity(query, ref_emb).item()
        scores.append((name, score))

    scores.sort(key=lambda x: x[1], reverse=True)

    best_name, best_score = scores[0]
    second_score = scores[1][1] if len(scores) > 1 else -1

    gap = best_score - second_score

    return best_name, best_score, gap, scores

start = perf_counter()
name, score, gap, top = find_item("aaaa.png")
end = perf_counter()

print(name, score, gap)
print(top)
print(f"Time taken: {end - start:.4f} seconds")