import torch
import clip
from PIL import Image
from time import perf_counter

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def encode_image(path):
    image = preprocess(Image.open(path)).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model.encode_image(image)

    return features / features.norm(dim=-1, keepdim=True)

reference_images = {
    "a": "../debug/wuwa_inventory_system_20.png/attempt_85/14_icon_0.png",
    "b": "../debug/wuwa_inventory_system_20.png/attempt_85/14_icon_1.png",
    "c": "../debug/wuwa_inventory_system_20.png/attempt_85/14_icon_2.png"
}

reference_features = {}

for name, path in reference_images.items():
    reference_features[name] = encode_image(path)

def find_best_match(image_path):
    query = encode_image(image_path)

    best_name = None
    best_score = -1

    for name, ref in reference_features.items():
        score = (query @ ref.T).item()   # cosine similarity

        if score > best_score:
            best_score = score
            best_name = name

    return best_name, best_score

result = []
for i in range(18):
    start = perf_counter()
    name, score = find_best_match(f"../items_assets/icons/{i}_i.png")
    end = perf_counter()

    result.append((i, name, score, end - start))

for i, name, score, time in sorted(result, key=lambda x: x[2], reverse=True):
    print(f"\nTesting ../items_assets/icons/{i}_i.png")
    print("Prediction:", name)
    print("Score:", score)
    print("Time taken:", time)