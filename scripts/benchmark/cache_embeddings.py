"""Precompute and cache image encoder embeddings for PT-SAM."""

import glob
import os
import time

import numpy as np
import torch
from tqdm import tqdm


def precompute_embeddings(sam_checkpoint, data_root, cache_dir, device="cuda:0"):
    """Run frozen SAM image encoder on all images, save embeddings to disk.

    Each embedding is (256, 64, 64) float32 ~ 4MB per file.
    For 160 images, total cache is ~640MB on disk.
    """
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "MedSAM-main"))
    from segment_anything import sam_model_registry

    os.makedirs(cache_dir, exist_ok=True)

    img_dir = os.path.join(data_root, "imgs")
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.npy")))
    print(f"Caching embeddings for {len(img_files)} images -> {cache_dir}")

    sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
    image_encoder = sam.image_encoder.to(device).eval()
    del sam.mask_decoder, sam.prompt_encoder
    torch.cuda.empty_cache()

    start = time.time()
    cached = 0
    skipped = 0

    with torch.no_grad():
        for img_file in tqdm(img_files, desc="Caching embeddings"):
            name = os.path.basename(img_file).replace(".npy", ".pt")
            out_path = os.path.join(cache_dir, name)

            if os.path.exists(out_path):
                skipped += 1
                continue

            img = np.load(img_file, allow_pickle=True)
            img_tensor = torch.tensor(img.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

            with torch.amp.autocast("cuda"):
                embedding = image_encoder(img_tensor)

            torch.save(embedding.cpu().squeeze(0), out_path)
            cached += 1

    elapsed = time.time() - start
    total = cached + skipped
    print(f"Done: {cached} computed, {skipped} already cached, {total} total")
    print(f"Time: {elapsed:.1f}s ({elapsed/max(cached,1):.2f}s per image)")

    # Verify
    cache_files = glob.glob(os.path.join(cache_dir, "*.pt"))
    assert len(cache_files) == len(img_files), (
        f"Cache mismatch: {len(cache_files)} cached vs {len(img_files)} images"
    )
    return cache_dir
