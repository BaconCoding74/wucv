from PIL import Image
from pathlib import Path
from typing import Literal
from transformers import AutoImageProcessor, AutoModel

import os
import torch
import numpy as np
import torch.nn.functional as F
import cv2
from hashlib import sha256

def process_image_input(image: str | Path | Image.Image | np.ndarray) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    
    if isinstance(image, np.ndarray):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
    else:
        image = Image.open(image).convert("RGB")
    
    return image

class DinoV2:
    def __init__(
        self, 
        model_name: str = "facebook/dinov2-base",
        preprocessor_version: str = "1.0",
    ):
        self.model_name = model_name
        self.processor_version = preprocessor_version
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

        self.model.eval()
        self.embedding_size = self.model.config.hidden_size

    @torch.inference_mode()
    def get_embedding(
        self, 
        images: list[str | Path | Image.Image | np.ndarray],
        return_patches: bool = False,
    ) -> tuple[torch.Tensor, None] | tuple[torch.Tensor, torch.Tensor]:
        images = [process_image_input(image) for image in images]

        inputs = self.processor(
            images=images, 
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**inputs)
        tokens = outputs.last_hidden_state

        cls_embeddings = tokens[:, 0, :]
        cls_embeddings = F.normalize(
            cls_embeddings,
            p=2,
            dim=-1,
        )

        if not return_patches:
            return cls_embeddings, None

        patches_embeddings = tokens[:, 1:, :]
        patches_embeddings = F.normalize(
            patches_embeddings,
            p=2,
            dim=-1,
        )

        return cls_embeddings, patches_embeddings

class EmbeddingStore:
    def __init__(
        self, 
        cache_path: str,
        input_dir: str,
        encoder: DinoV2,
    ):
        self.cache_path = Path(cache_path)
        self.input_dir = Path(input_dir)
        self.encoder = encoder
        self.valid_extensions = [
            ".jpg", 
            ".jpeg", 
            ".png",
        ]

        self.image_paths = []
        self.image_hashes = []
        self.image_sizes = []
        self.image_mtime = []
        self.cls_embeddings = torch.empty(0, self.encoder.embedding_size, dtype=torch.float32)

    @property
    def is_empty(self) -> bool:
        return len(self.image_paths) == 0 or self.cls_embeddings.shape[0] == 0

    def _hash_file(self, file_path: Path) -> str:
        with open(file_path, "rb") as f:
            hasher = sha256()

            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break

                hasher.update(chunk)

            return hasher.hexdigest()

    def save(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "model_name": self.encoder.model_name,
            "preprocessor_version": self.encoder.processor_version,
            "embedding_size": self.encoder.embedding_size,
            "image_paths": self.image_paths,
            "image_hashes": self.image_hashes,
            "image_sizes": self.image_sizes,
            "image_mtime": self.image_mtime,
            "cls_embeddings": self.cls_embeddings,
        }

        temp_path = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")

        torch.save(
            data,
            temp_path
        )

        os.replace(temp_path, self.cache_path)

    def load(self):
        if not self.cache_path.exists():
            raise ValueError(f"Cache file does not exist: {self.cache_path}")

        # weights_only means it is restrict loader with tensor instead of meaning there is only weights
        data = torch.load(
            self.cache_path,
            map_location="cpu",
            weights_only=True,
        )

        if (data["model_name"] != self.encoder.model_name):
            raise ValueError(
                f"Model name mismatch: {data['model_name']} != {self.encoder.model_name}"
            )

        if (data["preprocessor_version"] != self.encoder.processor_version):
            raise ValueError(
                f"Preprocessor version mismatch: {data['preprocessor_version']} != {self.encoder.processor_version}"
            )

        if (data["embedding_size"] != self.encoder.embedding_size):
            raise ValueError(
                f"Embedding size mismatch: {data['embedding_size']} != {self.encoder.embedding_size}"
            )

        if not isinstance(data["cls_embeddings"], torch.Tensor):
            raise ValueError(
                f"cls_embeddings is not a torch.Tensor: {type(data['cls_embeddings'])}"
            )

        self.image_paths = data["image_paths"]
        self.image_hashes = data["image_hashes"]
        self.image_sizes = data["image_sizes"]
        self.image_mtime = data["image_mtime"]
        self.cls_embeddings = data["cls_embeddings"].detach().cpu()

        return True

    def sync(
        self,
        batch_size: int = 32,
        force_hash_check: bool = False,
        rebuild_on_mismatch: bool = False,
    ):
        if not self.input_dir.exists():
            raise ValueError(f"Input directory does not exist: {self.input_dir}")

        image_paths: list[Path] = sorted(
            path for path in self.input_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in self.valid_extensions
        )

        try:
            self.load()
        except Exception as e:
            print(f"Failed to load cache: {e}")

            if not rebuild_on_mismatch:
                raise e
            print("Rebuilding cache...")

            self.image_paths = []
            self.image_hashes = []
            self.image_sizes = []
            self.image_mtime = []
            self.cls_embeddings = torch.empty(0, self.encoder.embedding_size, dtype=torch.float32)

        path_to_index = {path: index for index, path in enumerate(self.image_paths)}
        hash_to_index = {hash: index for index, hash in enumerate(self.image_hashes)}
        old_paths = set(self.image_paths)

        results = {}
        pending_paths = []
        pending_info = []
        current_relative_paths = set()

        stats = {
            "total": len(image_paths),
            "new": 0,
            "modified": 0,
            "unchanged": 0,
            "deleted": 0,
            "reused_hashes": 0,
            "embedded": 0,
        }

        for image_path in image_paths:
            relative_path = image_path.relative_to(self.input_dir).as_posix()
            current_relative_paths.add(relative_path)

            image_stats = image_path.stat()
            image_size = image_stats.st_size
            image_mtime = image_stats.st_mtime_ns

            cached_index = path_to_index.get(relative_path)
            # Sync operation if image path exists in the cache
            if cached_index is not None:
                cached_hash = self.image_hashes[cached_index]
                cached_size = self.image_sizes[cached_index]
                cached_mtime = self.image_mtime[cached_index]

                # Check image metadata to check if image had been changed
                if (cached_size == image_size) and (cached_mtime == image_mtime) and not force_hash_check:
                    results[relative_path] = {
                        "size": image_size,
                        "mtime": image_mtime,
                        "hash": cached_hash,
                        "embedding": self.cls_embeddings[cached_index],
                    }

                    stats["unchanged"] += 1
                    continue

                # If metadata is different or force hash check, check if image hash is the same
                cached_hash = self.image_hashes[cached_index]
                current_hash = self._hash_file(image_path)

                if (cached_hash == current_hash):
                    results[relative_path] = {
                        "size": image_size,
                        "mtime": image_mtime,
                        "hash": cached_hash,
                        "embedding": self.cls_embeddings[cached_index],
                    }

                    stats["unchanged"] += 1
                    continue

                # If the hash is different, check if the hash exists in the cache
                stats["modified"] += 1
                same_hash_index = hash_to_index.get(current_hash)

                if same_hash_index is not None:
                    results[relative_path] = {
                        "size": image_size,
                        "mtime": image_mtime,
                        "hash": current_hash,
                        "embedding": self.cls_embeddings[same_hash_index],
                    }

                    stats["reused_hashes"] += 1
                    continue

                # If there is not matching hash in cahce, we need to embed the image
                pending_paths.append(image_path)
                pending_info.append((relative_path, image_size, image_mtime, current_hash))
                continue

            # If image path is new to cache, check if the hash exists in the cache
            stats["new"] += 1

            current_hash = self._hash_file(image_path)
            same_hash_index = hash_to_index.get(current_hash)

            if same_hash_index is not None:
                results[relative_path] = {
                    "size": image_size,
                    "mtime": image_mtime,
                    "hash": current_hash,
                    "embedding": self.cls_embeddings[same_hash_index],
                }

                stats["reused_hashes"] += 1
                continue

            # If there is not matching hash in cahce, we need to embed the image
            pending_paths.append(image_path)
            pending_info.append((relative_path, image_size, image_mtime, current_hash))

        stats["deleted"] = len(old_paths - current_relative_paths)

        # Generate embedding for pending images
        for start in range(0, len(pending_paths), batch_size):
            end = min(start + batch_size, len(pending_paths))
            batch_image_paths = pending_paths[start:end]
            batch_info = pending_info[start:end]

            batch_cls_embeddings, _ = self.encoder.get_embedding(batch_image_paths)
            batch_cls_embeddings = batch_cls_embeddings.detach().cpu()

            for (relative_path, image_size, image_mtime, current_hash), embedding in zip(batch_info, batch_cls_embeddings):
                results[relative_path] = {
                    "size": image_size,
                    "mtime": image_mtime,
                    "hash": current_hash,
                    "embedding": embedding,
                }

                stats["embedded"] += 1

        final_image_paths = sorted(results.keys())

        self.image_paths = final_image_paths
        self.image_hashes = [results[path]["hash"] for path in final_image_paths]
        self.image_sizes = [results[path]["size"] for path in final_image_paths]
        self.image_mtime = [results[path]["mtime"] for path in final_image_paths]

        if final_image_paths:
            self.cls_embeddings = torch.stack([results[path]["embedding"] for path in final_image_paths])
        else:
            self.cls_embeddings = torch.empty(0, self.encoder.embedding_size, dtype=torch.float32)

        self.save()
        return stats

class ImageRetriever:
    def __init__(
        self,
        encoder: DinoV2,
        embedding_store: EmbeddingStore,
        device_mode: Literal["cuda", "cpu", "auto"] = "auto",
        query_batch_size: int = 32,
        reserved_vram: float = 2.0,
        safety_factor: float = 1.2,
    ):
        self.encoder = encoder
        self.embedding_store = embedding_store
        self.device_mode = device_mode
        self.query_batch_size = query_batch_size
        self.reserved_vram = reserved_vram
        self.safety_factor = safety_factor

        self.refresh()

    def refresh(self):
        if self.device_mode == "auto":
            self.retrieval_device = self._choose_device()
        else:
            self.retrieval_device = torch.device(self.device_mode)

        self.reference_paths = self.embedding_store.image_paths.copy()
        self.reference_embeddings = self.embedding_store.cls_embeddings.to(self.retrieval_device)

    def _choose_device(self):
        if not torch.cuda.is_available():
            self.retrieval_device = torch.device("cpu")
            return

        # Get embedding size
        n, d = self.embedding_store.cls_embeddings.shape
        element_size = self.embedding_store.cls_embeddings.element_size()


        reference_bytes = self.embedding_store.cls_embeddings.nbytes
        query_bytes = self.query_batch_size * d * element_size
        score_bytes = self.query_batch_size * n * element_size

        required_bytes = reference_bytes + query_bytes + score_bytes
        required_bytes *= self.safety_factor

        reserved_bytes = self.reserved_vram * 1024**3

        free_bytes, _ = torch.cuda.mem_get_info()

        if required_bytes + reserved_bytes > free_bytes:
            return torch.device("cpu")
        else:
            return torch.device("cuda")

    def retrieve_top_k(
        self, 
        input_paths: list[str | Path | Image.Image | np.ndarray],
        k: int = 5,
    ) -> list[list[dict]]:
        query_embedding, _ = self.encoder.get_embedding(input_paths)
        scores = query_embedding @ self.reference_embeddings.to(query_embedding.device).T

        k = min(k, scores.shape[1])
        top_scores, top_indices = torch.topk(scores, k=k)
        results = []

        for query_scores, query_indices in zip(top_scores, top_indices):
            temp = []

            for top_score, top_index in zip(query_scores, query_indices):
                temp.append({
                    "retrieved_path": self.reference_paths[top_index],
                    "retrieved_index": top_index.item(),
                    "score": top_score.item(),
                })

            results.append(temp)

        return results


def test():
    input_dir = Path("image_recognition/datasets/datasets_v002")

    encoder = DinoV2(
        model_name="facebook/dinov2-small", 
        preprocessor_version="1.0",
    )

    embedding_store = EmbeddingStore(
        cache_path="cache/dinov2_small_cache.pt", 
        input_dir=str(input_dir), 
        encoder=encoder,
    )

    stats = embedding_store.sync(
        batch_size=32, 
        force_hash_check=False, 
        rebuild_on_mismatch=True,
    )

    retriever = ImageRetriever(
        encoder=encoder,
        embedding_store=embedding_store,
        device_mode="auto",
        query_batch_size=32,
        reserved_vram=2.0,
        safety_factor=1.2,
    )

    input_image_paths = [
        path 
        for path in input_dir.rglob("*") 
        if path.is_file() and path.suffix.lower() in embedding_store.valid_extensions
    ]

    retrieved_results = retriever.retrieve_top_k(
        input_paths=input_image_paths,
    )

    import cv2
    from skimage.metrics import structural_similarity as ssim

    for input_path, results in zip(input_image_paths, retrieved_results):
        print(f"Input image: {input_path}")

        query_image = cv2.imread(str(input_path))
        query_image = cv2.resize(query_image, (128, 128), interpolation=cv2.INTER_AREA)

        for result in results:
            color_code = "\033[92m"

            retrieved_image = cv2.imread(str(input_dir / result["retrieved_path"]))
            retrieved_image = cv2.resize(retrieved_image, (128, 128), interpolation=cv2.INTER_AREA)

            # Sensitive to pixel alignment and geometry
            ssim_score, ssim_map = ssim(query_image, retrieved_image, channel_axis=-1, data_range=255, full=True)

            # Pooling channel score
            if ssim_map.ndim == 3:
                ssim_map = ssim_map.mean(axis=-1)

            # Create masking for new indicator
            h, w = ssim_map.shape[:2]
            valid_mask = np.ones((h, w), dtype=bool)
            valid_mask[:int(h * 0.235), int(w * 0.781):] = False

            # Calculate masked SSIM score
            masked_ssim_score = ssim_map[valid_mask].mean()

            # If query and retrieved image are from different folders, print in red
            if input_path.parent.name != Path(result["retrieved_path"]).parent.name:
                color_code = "\033[31m"

            print(f"{color_code}\tRetrieved image: {result['retrieved_path']}, Score: {result['score']:.4f}, SSIM: {ssim_score:.4f}, Masked SSIM: {masked_ssim_score:.4f}\033[0m")