import random
import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader

# =============================================================================
# TWEAK VARIABLES — adjust these to tune training
# =============================================================================

# --- Data ---
DATASET_PATH        = "datasets"        # Root folder; subfolders = classes
SAMPLES_PER_EPOCH   = 5000              # Virtual epoch size (random sampling)
BATCH_SIZE          = 64               # Larger = more hard pairs per batch

# --- Model ---
EMBEDDING_DIM       = 128              # Size of output feature vector (64/128/256)

# --- Loss ---
MARGIN              = 0.3              # Triplet margin; lower = tighter clusters

# --- Optimizer ---
LEARNING_RATE       = 1e-4             # Initial LR for Adam
WEIGHT_DECAY        = 1e-4             # L2 regularisation to reduce overfitting

# --- Scheduler ---
NUM_EPOCHS          = 30              # Total training epochs
LR_MIN              = 1e-6             # Minimum LR at end of cosine schedule

# --- Augmentation ---
RESIZE_TO           = 160              # Resize before crop
CROP_TO             = 128              # Final input resolution
CROP_SCALE_MIN      = 0.35             # Min scale for RandomResizedCrop
CROP_SCALE_MAX      = 1.0              # Max scale for RandomResizedCrop
COLOR_JITTER_PROB   = 0.8              # Probability of applying colour jitter
BRIGHTNESS          = 0.3
CONTRAST            = 0.3
SATURATION          = 0.25
HUE                 = 0.05
GRAYSCALE_PROB      = 0.1              # Forces model to learn shape over colour
AFFINE_DEGREES      = 5
AFFINE_TRANSLATE    = (0.08, 0.08)
AFFINE_SCALE        = (0.9, 1.15)

# --- Output ---
SAVE_PATH           = "item_recognition.pth"

# --- Workers ---
# Windows does not support forked workers — keep this at 0
# On Linux/Mac you can raise this to 2-4 for faster data loading
NUM_WORKERS         = 0

# =============================================================================


device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Augmentation pipeline
# ---------------------------------------------------------------------------

tf = transforms.Compose([
    transforms.Resize((RESIZE_TO, RESIZE_TO)),
    transforms.RandomResizedCrop(
        CROP_TO,
        scale=(CROP_SCALE_MIN, CROP_SCALE_MAX),
        ratio=(0.8, 1.25),
    ),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([
        transforms.ColorJitter(
            brightness=BRIGHTNESS,
            contrast=CONTRAST,
            saturation=SATURATION,
            hue=HUE,
        ),
    ], p=COLOR_JITTER_PROB),
    transforms.RandomGrayscale(p=GRAYSCALE_PROB),
    transforms.RandomAffine(
        degrees=AFFINE_DEGREES,
        translate=AFFINE_TRANSLATE,
        scale=AFFINE_SCALE,
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Inference transform (no augmentation)
tf_infer = transforms.Compose([
    transforms.Resize((CROP_TO, CROP_TO)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ItemDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform or tf

        class_dirs = sorted([p for p in self.root.iterdir() if p.is_dir()])
        self.label_names = [p.name for p in class_dirs]
        self.label_to_idx = {n: i for i, n in enumerate(self.label_names)}

        self.samples = []
        for class_dir in class_dirs:
            idx = self.label_to_idx[class_dir.name]
            for img_path in class_dir.glob("*.*"):
                self.samples.append((img_path, idx))

        print(f"Dataset: {len(self.label_names)} classes, "
              f"{len(self.samples)} images total")

    def __len__(self):
        return SAMPLES_PER_EPOCH

    def __getitem__(self, _):
        img_path, label = random.choice(self.samples)
        img = Image.open(img_path).convert("RGB")
        return self.transform(img), label


# ---------------------------------------------------------------------------
# Embedding network
# ---------------------------------------------------------------------------

class EmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        base = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT
        )
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embedding = nn.Sequential(
            nn.Linear(576, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.embedding(x)
        return nn.functional.normalize(x, p=2, dim=1)


# ---------------------------------------------------------------------------
# Online hard triplet mining loss
# ---------------------------------------------------------------------------

def hard_triplet_loss(embeddings, labels, margin=MARGIN):
    dist = torch.cdist(embeddings, embeddings, p=2)

    loss_sum = torch.tensor(0.0, device=embeddings.device, requires_grad=True)
    valid = 0

    for i in range(len(labels)):
        same = (labels == labels[i])
        same[i] = False
        diff = (labels != labels[i])

        if same.sum() == 0 or diff.sum() == 0:
            continue

        hardest_pos = dist[i][same].max()
        hardest_neg = dist[i][diff].min()

        triplet_loss = torch.clamp(hardest_pos - hardest_neg + margin, min=0.0)
        loss_sum = loss_sum + triplet_loss
        valid += 1

    return loss_sum / max(valid, 1)


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

class ImageIdentifier:
    """
    Build a gallery of known embeddings, then identify query images by
    nearest neighbour in embedding space.

    Usage:
        identifier = ImageIdentifier("item_recognition.pth", "datasets")
        identifier.build_gallery()
        results = identifier.identify("query.jpg")
        for label, score in results:
            print(f"{label}: {score:.3f}")
    """
    def __init__(self, model_path, dataset_root):
        self.model = EmbeddingNet().to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.dataset_root = Path(dataset_root)
        self.gallery_embeddings = None
        self.gallery_labels = []

    @torch.no_grad()
    def _embed(self, img_path):
        img = Image.open(img_path).convert("RGB")
        tensor = tf_infer(img).unsqueeze(0).to(device)
        return self.model(tensor)

    def build_gallery(self, samples_per_class=5):
        all_embeddings = []
        all_labels = []
        for class_dir in sorted(self.dataset_root.iterdir()):
            if not class_dir.is_dir():
                continue
            imgs = list(class_dir.glob("*.*"))[:samples_per_class]
            if not imgs:
                continue
            class_embs = torch.cat([self._embed(p) for p in imgs], dim=0)
            mean_emb = nn.functional.normalize(
                class_embs.mean(dim=0, keepdim=True), p=2, dim=1
            )
            all_embeddings.append(mean_emb)
            all_labels.append(class_dir.name)

        self.gallery_embeddings = torch.cat(all_embeddings, dim=0)
        self.gallery_labels = all_labels
        print(f"Gallery built: {len(all_labels)} classes")

    @torch.no_grad()
    def identify(self, img_path, top_k=3):
        if self.gallery_embeddings is None:
            raise RuntimeError("Call build_gallery() first.")
        query = self._embed(img_path)
        sims = (self.gallery_embeddings @ query.T).squeeze(1)
        top = sims.topk(min(top_k, len(self.gallery_labels)))
        return [(self.gallery_labels[i], sims[i].item())
                for i in top.indices.tolist()]


# ---------------------------------------------------------------------------
# Entry point — required on Windows to avoid multiprocessing crash
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Using device: {device}")
    if device == "cpu":
        print(
            "\n  WARNING: Running on CPU. Training will be very slow.\n"
            "  To enable GPU, install the CUDA build of PyTorch:\n"
            "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121\n"
            "  (replace cu121 with your CUDA version — check with: nvidia-smi)\n"
        )

    dataset = ItemDataset(DATASET_PATH)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device == "cuda"),
    )

    model = EmbeddingNet().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=LR_MIN
    )

    best_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        batches = 0

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            embeddings = model(images)
            loss = hard_triplet_loss(embeddings, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        scheduler.step()
        avg_loss = total_loss / max(batches, 1)
        lr_now = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1:>3}/{NUM_EPOCHS}  loss={avg_loss:.4f}  lr={lr_now:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  ✓ Saved new best model (loss={best_loss:.4f})")

    print(f"\nDone. Best loss: {best_loss:.4f}  →  {SAVE_PATH}")

    # Uncomment to test inference after training:
    # identifier = ImageIdentifier(SAVE_PATH, DATASET_PATH)
    # identifier.build_gallery()
    # results = identifier.identify("query.jpg")
    # for label, score in results:
    #     print(f"  {label}: {score:.3f}")