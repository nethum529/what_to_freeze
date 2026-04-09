"""
Three SAM freeze strategies for the benchmark.

- MedSAM:  freeze prompt_encoder, train image_encoder + mask_decoder
- PP-SAM:  freeze mask_decoder, train image_encoder + prompt_encoder
- PT-SAM:  freeze everything, learn 8 prompt tokens (2,048 params)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint


def _encode_image_with_checkpointing(image_encoder, x):
    """Run image encoder with gradient checkpointing on transformer blocks."""
    x = image_encoder.patch_embed(x)
    if image_encoder.pos_embed is not None:
        x = x + image_encoder.pos_embed
    for blk in image_encoder.blocks:
        x = grad_checkpoint(blk, x, use_reentrant=False)
    x = image_encoder.neck(x.permute(0, 3, 1, 2))
    return x


def _encode_image_direct(image_encoder, x):
    """Run image encoder without gradient checkpointing."""
    return image_encoder(x)


class MedSAMStrategy(nn.Module):
    """Freeze prompt encoder. Train image encoder + mask decoder."""

    def __init__(self, image_encoder, mask_decoder, prompt_encoder,
                 use_grad_checkpoint=True):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.use_grad_checkpoint = use_grad_checkpoint

        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, box):
        if self.use_grad_checkpoint and self.training:
            image_embedding = _encode_image_with_checkpointing(
                self.image_encoder, image
            )
        else:
            image_embedding = _encode_image_direct(self.image_encoder, image)

        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None, boxes=box_torch, masks=None,
            )

        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        ori_res_masks = F.interpolate(
            low_res_masks, size=(image.shape[2], image.shape[3]),
            mode="bilinear", align_corners=False,
        )
        return ori_res_masks

    def get_trainable_params(self):
        return (
            list(self.image_encoder.parameters())
            + list(self.mask_decoder.parameters())
        )


class PPSAMStrategy(nn.Module):
    """Freeze mask decoder. Train image encoder + prompt encoder."""

    def __init__(self, image_encoder, mask_decoder, prompt_encoder,
                 use_grad_checkpoint=True):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.use_grad_checkpoint = use_grad_checkpoint

        for param in self.mask_decoder.parameters():
            param.requires_grad = False

    def forward(self, image, box):
        if self.use_grad_checkpoint and self.training:
            image_embedding = _encode_image_with_checkpointing(
                self.image_encoder, image
            )
        else:
            image_embedding = _encode_image_direct(self.image_encoder, image)

        box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
        if len(box_torch.shape) == 2:
            box_torch = box_torch[:, None, :]
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None, boxes=box_torch, masks=None,
        )

        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        ori_res_masks = F.interpolate(
            low_res_masks, size=(image.shape[2], image.shape[3]),
            mode="bilinear", align_corners=False,
        )
        return ori_res_masks

    def get_trainable_params(self):
        return (
            list(self.image_encoder.parameters())
            + list(self.prompt_encoder.parameters())
        )


class PTSAMStrategy(nn.Module):
    """Freeze everything. Learn 8 prompt tokens (8 x 256 = 2,048 params)."""

    def __init__(self, image_encoder, mask_decoder, prompt_encoder,
                 num_tokens=8, use_grad_checkpoint=False):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

        for param in self.image_encoder.parameters():
            param.requires_grad = False
        for param in self.mask_decoder.parameters():
            param.requires_grad = False
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

        embed_dim = prompt_encoder.embed_dim  # 256
        self.learned_tokens = nn.Parameter(torch.randn(num_tokens, embed_dim) * 0.02)

    def forward(self, image=None, box=None, precomputed_embedding=None):
        if precomputed_embedding is not None:
            image_embedding = precomputed_embedding
        else:
            with torch.no_grad():
                image_embedding = self.image_encoder(image)

        B = image_embedding.shape[0]
        sparse_embeddings = self.learned_tokens.unsqueeze(0).expand(B, -1, -1)

        with torch.no_grad():
            dense_embeddings = self.prompt_encoder.no_mask_embed.weight.reshape(
                1, -1, 1, 1
            ).expand(
                B, -1,
                self.prompt_encoder.image_embedding_size[0],
                self.prompt_encoder.image_embedding_size[1],
            )

        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        if image is not None:
            h, w = image.shape[2], image.shape[3]
        else:
            h, w = 1024, 1024

        ori_res_masks = F.interpolate(
            low_res_masks, size=(h, w),
            mode="bilinear", align_corners=False,
        )
        return ori_res_masks

    def get_trainable_params(self):
        return [self.learned_tokens]


class BaseSAMStrategy(nn.Module):
    """Zero-shot SAM baseline. Freeze everything, 0 trainable params."""

    def __init__(self, image_encoder, mask_decoder, prompt_encoder,
                 use_grad_checkpoint=False):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, image, box):
        with torch.no_grad():
            image_embedding = self.image_encoder(image)

            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None, boxes=box_torch, masks=None,
            )

            low_res_masks, _ = self.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

        ori_res_masks = F.interpolate(
            low_res_masks, size=(image.shape[2], image.shape[3]),
            mode="bilinear", align_corners=False,
        )
        return ori_res_masks

    def get_trainable_params(self):
        return []


STRATEGY_REGISTRY = {
    "basesam": (BaseSAMStrategy, 20),
    "medsam": (MedSAMStrategy, 20),
    "ppsam": (PPSAMStrategy, 50),
    "ptsam": (PTSAMStrategy, 20),
}


def build_strategy(strategy_name, sam_model, config):
    """Build a strategy module from a loaded SAM model."""
    use_gc = config.GRADIENT_CHECKPOINT and strategy_name not in ("ptsam", "basesam")
    kwargs = dict(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    )
    if strategy_name == "ptsam":
        kwargs["num_tokens"] = config.NUM_PROMPT_TOKENS
    kwargs["use_grad_checkpoint"] = use_gc
    strategy_cls = STRATEGY_REGISTRY[strategy_name][0]
    return strategy_cls(**kwargs)
