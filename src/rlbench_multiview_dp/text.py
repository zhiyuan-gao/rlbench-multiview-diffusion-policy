from typing import Sequence

import numpy as np
import torch


@torch.inference_mode()
def encode_clip_texts(
    texts: Sequence[str],
    model_name: str = "openai/clip-vit-large-patch14",
    device: str = "cuda",
    batch_size: int = 64,
    local_files_only: bool = False,
    normalize: bool = True,
) -> np.ndarray:
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    tokenizer = CLIPTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    model = CLIPTextModelWithProjection.from_pretrained(model_name, local_files_only=local_files_only)
    model = model.to(device)
    model.eval()
    outputs = []
    for start in range(0, len(texts), int(batch_size)):
        batch = list(texts[start : start + int(batch_size)])
        tokens = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        feats = model(**tokens).text_embeds
        if normalize:
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        outputs.append(feats.detach().cpu().float().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32)


def dummy_text_features(texts: Sequence[str], dim: int = 512, seed: int = 0) -> np.ndarray:
    import hashlib

    features = []
    for text in texts:
        digest = hashlib.sha256(f"{seed}:{text}".encode("utf-8")).digest()
        local_seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
        rng = np.random.default_rng(local_seed)
        vec = rng.standard_normal(int(dim)).astype(np.float32)
        vec /= max(float(np.linalg.norm(vec)), 1e-12)
        features.append(vec)
    return np.stack(features, axis=0).astype(np.float32)

