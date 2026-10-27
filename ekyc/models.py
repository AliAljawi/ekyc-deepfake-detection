"""
The detectors: CLIP ViT-B/16 + LoRA (main), frozen CLIP + linear probe (embedding extraction), EfficientNet-B4
(baseline), and zero-shot CLIP (text prompts, no training). Every model takes uint8 (B, 3, 256, 256) images and
returns one fp32 logit per image (higher = more likely fake).

Shared rules: one preprocess() for training, scoring, heat maps and the demo; features run under bf16 autocast,
and the final linear layer runs in fp32 with autocast off. Checkpoints hold only what training changed (LoRA + head;
the probe's head; all of EfficientNet, including its BatchNorm statistics) as safetensors plus a JSON sidecar.
"""
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ekyc.config import CLIP_ID, EFFNET_ID, SPEC
from ekyc.utils import atomic_write

KINDS = ("clip_lora", "clip_probe", "effnet_b4")
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj"]
EXPECTED_TRAINABLE = {"clip_lora": 1_179_648 + 769, "clip_probe": 769}  # EfficientNet: every parameter
REAL_PROMPTS = ("a photo of a real human face", "a real photograph of a person")
FAKE_PROMPTS = ("a deepfake image of a face", "a fake, AI-generated face")


def preprocess(x_u8, spec):
    """uint8 (B, 3, 256, 256) -> float normalised for the model: CLIP gets a bicubic antialiased resize to 224 and
    CLIP's mean/std; EfficientNet stays at 256 with ImageNet's mean/std (channels-last for speed)."""
    x = x_u8.float().div_(255)
    if x.shape[-1] != spec.size:
        x = F.interpolate(x, size=(spec.size, spec.size), mode="bicubic", antialias=True,
                          align_corners=False).clamp_(0, 1)
    mean = torch.tensor(spec.mean, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(spec.std, device=x.device).view(1, 3, 1, 1)
    x = (x - mean) / std
    return x.contiguous(memory_format=torch.channels_last) if spec.channels_last else x


def _clip_vision():
    from transformers import CLIPVisionModel
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()  # hides the harmless "unexpected text_model.* keys" report
    return CLIPVisionModel.from_pretrained(CLIP_ID, attn_implementation="sdpa")


class Detector(nn.Module):
    def __init__(self, kind, backbone=None, head=None, net=None):
        super().__init__()
        self.kind, self.spec = kind, SPEC[kind]
        if kind == "effnet_b4":
            self.net = net
        else:
            self.backbone, self.head = backbone, head

    def features(self, x_u8):
        """The representation the final linear layer reads: CLIP's pooled CLS token (768) or EfficientNet's
        globally pooled features (1792), in fp32."""
        x = preprocess(x_u8, self.spec)
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.is_cuda):
            if self.kind == "effnet_b4":
                return self.net.forward_head(self.net.forward_features(x), pre_logits=True).float()
            return self.backbone(pixel_values=x).pooler_output.float()

    def forward(self, x_u8):
        x = preprocess(x_u8, self.spec)
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.is_cuda):
            if self.kind == "effnet_b4":
                f = self.net.forward_features(x)
            elif self.kind == "clip_probe":
                with torch.no_grad():
                    f = self.backbone(pixel_values=x).pooler_output
            else:
                f = self.backbone(pixel_values=x).pooler_output
        with torch.autocast(x.device.type, enabled=False):
            out = self.net.forward_head(f.float()) if self.kind == "effnet_b4" else self.head(f.float())
        return out.squeeze(1)

    def train(self, mode=True):
        super().train(mode)
        if self.kind == "clip_probe":
            self.backbone.eval()  # frozen: always in evaluation mode
        return self

    def is_head(self, name):
        return name.startswith(("head.", "net.classifier."))

    def param_groups(self, lr, head_lr, wd):
        """AdamW groups: weight decay only on matrices (not biases or norms); the new head gets head_lr, except
        for the probe, whose head is the whole trainable model and gets lr."""
        groups = {}
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            head = self.is_head(name)
            rate = head_lr if head and self.kind != "clip_probe" else lr
            groups.setdefault((head, rate, wd if p.ndim >= 2 else 0.0), []).append(p)
        return [{"params": ps, "lr": r, "weight_decay": d, "name": ("head" if h else "body") + ("" if d else "_no_wd")}
                for (h, r, d), ps in groups.items()]

    def n_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def checkpoint_state(self):
        """Only what training changes: LoRA + head, the probe's head, or all of EfficientNet (incl. BatchNorm)."""
        sd = self.state_dict()
        if self.kind == "effnet_b4":
            keep = sd
        elif self.kind == "clip_lora":
            keep = {k: v for k, v in sd.items() if "lora_" in k or k.startswith("head.")}
        else:
            keep = {k: v for k, v in sd.items() if k.startswith("head.")}
        return {k: v.detach().contiguous().cpu() for k, v in keep.items()}


def build_model(kind, seed=0):
    """A fresh model. The seed fixes the random parts (LoRA's A matrices and the new head)."""
    if kind not in KINDS:
        raise ValueError(f"unknown model kind {kind!r}")
    torch.manual_seed(seed)
    if kind == "effnet_b4":
        import timm
        net = timm.create_model(EFFNET_ID, pretrained=True, num_classes=1, drop_rate=0.0, drop_path_rate=0.0)
        # timm initialises a new 1-output classifier with U(-1, 1) (TF-style, 1/sqrt(fan_out)); give it PyTorch's
        # default Linear initialisation instead, like the CLIP heads (and DeepfakeBench's EfficientNet-B4 head)
        net.classifier.reset_parameters()
        return Detector(kind, net=net.to(memory_format=torch.channels_last))
    clip = _clip_vision()
    if kind == "clip_lora":
        from peft import LoraConfig, get_peft_model
        clip = get_peft_model(clip, LoraConfig(r=16, lora_alpha=16, lora_dropout=0.1, target_modules=LORA_TARGETS,
                                               bias="none"))
    else:
        clip.requires_grad_(False)
    model = Detector(kind, backbone=clip, head=nn.Linear(768, 1))
    if model.n_trainable() != EXPECTED_TRAINABLE[kind]:
        raise AssertionError(f"{kind}: {model.n_trainable():,} trainable parameters, "
                             f"expected {EXPECTED_TRAINABLE[kind]:,}")
    return model


def save_checkpoint(model, path, meta):
    from safetensors.torch import save
    path = Path(path)
    atomic_write(path, save(model.checkpoint_state()))
    atomic_write(path.with_suffix(".json"), json.dumps({"kind": model.kind, **meta}, indent=1))


def load_checkpoint(path, device="cuda"):
    """Rebuilds the model and loads the checkpoint; only frozen CLIP weights may be absent from the file."""
    from safetensors.torch import load_file
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    model = build_model(meta["kind"], seed=meta.get("seed", 0))
    state = load_file(path)
    if model.kind == "effnet_b4":
        model.load_state_dict(state, strict=True)
    else:
        missing, unexpected = model.load_state_dict(state, strict=False)
        trained = {k for k in model.state_dict() if "lora_" in k or k.startswith("head.")}
        if model.kind == "clip_probe":
            trained = {k for k in trained if k.startswith("head.")}
        if unexpected or trained & set(missing):
            raise AssertionError(f"{path.name}: unexpected keys {unexpected[:5]}, missing trained keys "
                                 f"{sorted(trained & set(missing))[:5]}")
    return model.to(device).eval(), meta


def set_eager_attention(model):
    """sdpa attention returns no attention maps; attention rollout needs the eager implementation."""
    model.backbone.set_attn_implementation("eager")


class ZeroShotCLIP(nn.Module):
    """CLIP with text prompts, no training: logit = CLIP's logit scale x (mean cosine similarity to the fake
    prompts - mean cosine similarity to the real prompts). The prompts were fixed before scoring.
    The text embeddings are computed once (the tokenizer files need a one-time download) and cached."""

    kind = "clip_zeroshot"

    def __init__(self, cache_dir):
        super().__init__()
        from transformers import CLIPModel
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()
        clip = CLIPModel.from_pretrained(CLIP_ID, attn_implementation="sdpa")
        self.vision, self.projection = clip.vision_model, clip.visual_projection
        self.spec = SPEC["clip_zeroshot"]
        cache = Path(cache_dir) / "zeroshot_text.pt"
        text = torch.load(cache) if cache.is_file() else None
        if text is not None and (tuple(text["prompts"]["real"]), tuple(text["prompts"]["fake"])) != (REAL_PROMPTS,
                                                                                                    FAKE_PROMPTS):
            text = None  # the prompts changed since the cache was written: recompute
        if text is None:
            from transformers import CLIPTokenizer
            tok = CLIPTokenizer.from_pretrained(CLIP_ID)
            with torch.no_grad():
                emb = {}
                for name, prompts in (("real", REAL_PROMPTS), ("fake", FAKE_PROMPTS)):
                    t = clip.get_text_features(**tok(list(prompts), padding=True, return_tensors="pt"))
                    t = t.pooler_output if hasattr(t, "pooler_output") else t
                    emb[name] = F.normalize(t, dim=-1)
            text = {"real": emb["real"], "fake": emb["fake"], "logit_scale": clip.logit_scale.exp().detach(),
                    "prompts": {"real": REAL_PROMPTS, "fake": FAKE_PROMPTS}}
            cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(text, cache)
        self.register_buffer("text_real", text["real"], persistent=False)
        self.register_buffer("text_fake", text["fake"], persistent=False)
        self.logit_scale = float(text["logit_scale"])
        self.prompts = {"real": list(REAL_PROMPTS), "fake": list(FAKE_PROMPTS)}
        del clip.text_model

    def forward(self, x_u8):
        x = preprocess(x_u8, self.spec)
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.is_cuda):
            f = self.vision(pixel_values=x).pooler_output
        with torch.autocast(x.device.type, enabled=False):
            img = F.normalize(self.projection(f.float()), dim=-1)
            return self.logit_scale * ((img @ self.text_fake.T).mean(1) - (img @ self.text_real.T).mean(1))
