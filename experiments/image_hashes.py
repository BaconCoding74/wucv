from PIL import Image
import imagehash

def compute_hash_distance(path1, path2):
    img1 = Image.open(path1)
    img2 = Image.open(path2)

    hash1 = imagehash.phash(img1)
    hash2 = imagehash.phash(img2)

    distance = hash1 - hash2
    return distance

def perpetual_hash_distance(paths):
    checked = []
    for path in paths:
        for other_path in paths:
            if path == other_path or other_path in checked:
                continue

            distance = compute_hash_distance(path, other_path)
            print(f"Distance between {path} and {other_path}: {distance}")
        checked.append(path)
        print()

paths = [
    "reference_items/e/4_i.png",
    "reference_items/e/4_a.png",
    "reference_items/c/2_i.png"
]

perpetual_hash_distance(paths)