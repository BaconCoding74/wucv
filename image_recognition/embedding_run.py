import os

import torch
import ir_constants as irc
from PIL import Image
from pathlib import Path
from torchvision import transforms
from time import perf_counter
from model_class import EmbeddingNet

device = "cuda" if torch.cuda.is_available() else "cpu"

model = EmbeddingNet().to(device)
model.load_state_dict(torch.load(f"{irc.MODEL_PATH}/{irc.SELECTED_MODEL}", map_location=device))
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

for item_folder in Path(irc.TEST_REFERENCES_PATH).iterdir():
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


for img in os.listdir(irc.TEST_INPUTS_PATH):
    start = perf_counter()
    name, score, gap, top = find_item(f"{irc.TEST_INPUTS_PATH}/{img}")
    end = perf_counter()

    print(f"Query: {img}")
    print(name, score, gap)
    print(top)
    print(f"Time taken: {end - start:.4f} seconds\n")
    