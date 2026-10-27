"""
Real or fake? The in-class demo for the Applied Vision Transformer Report.

Pick one of the 8 prepared faces and an image quality; each detector then says whether it accepts the face as
genuine at its fixed threshold (set so 1% of genuine FF++ validation faces are rejected, as in the report).
Runs on a CPU or a GPU, in Colab or on a laptop, from the folder written by export_demo.py (no internet needed
once the packages are installed).

  python demo_app.py                       # from inside that folder; opens http://127.0.0.1:7860
  python demo_app.py --dir <folder>        # from anywhere
  python demo_app.py --share               # also prints a public link
In a notebook:  import demo_app; demo_app.build("<folder>").launch()
"""
import argparse
import io
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

QUALITIES = {"Original": None, "JPEG 90": 90, "JPEG 75": 75, "JPEG 60": 60, "JPEG 45": 45, "JPEG 30": 30}
RED, TEAL = "#C8283C", "#1F9E89"


def compress(rgb, quality):
    """JPEG round trip at the given quality (None = the original PNG), as in the report's compression test."""
    if quality is None:
        return rgb
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"))


class Detectors:
    """The two exported detectors, run in fp32 without autocast: the path export_demo.py checked."""

    def __init__(self, folder):
        import timm
        from safetensors.torch import load_file
        from transformers import CLIPVisionConfig, CLIPVisionModel
        from transformers import logging as hf_logging

        hf_logging.set_verbosity_error()
        self.dir = Path(folder)
        self.cfg = json.loads((self.dir / "config.json").read_text(encoding="utf-8"))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        v, e = self.cfg["vit"], self.cfg["effnet"]
        sd = load_file(str(self.dir / v["file"]))
        self.vit = CLIPVisionModel(CLIPVisionConfig.from_dict(v["clip_vision_config"]))
        self.vit.load_state_dict({k[len("backbone."):]: t for k, t in sd.items() if k.startswith("backbone.")},
                                 strict=True)
        self.head = torch.nn.Linear(self.vit.config.hidden_size, 1)
        self.head.load_state_dict({k[len("head."):]: t for k, t in sd.items() if k.startswith("head.")}, strict=True)
        self.eff = timm.create_model(e["timm_name"], pretrained=False, num_classes=1)
        self.eff.load_state_dict(load_file(str(self.dir / e["file"])), strict=True)
        for m in (self.vit, self.head, self.eff):
            m.to(self.device).eval()

    @staticmethod
    def _normalise(x, size, mean, std):
        if x.shape[-1] != size:
            x = F.interpolate(x, size=(size, size), mode="bicubic", antialias=True, align_corners=False).clamp(0, 1)
        mean = torch.tensor(mean, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(std, device=x.device).view(1, 3, 1, 1)
        return (x - mean) / std

    @torch.no_grad()
    def logits(self, rgb):
        """uint8 (256, 256, 3) -> {"vit": logit, "effnet": logit}; higher means more likely fake."""
        x = torch.from_numpy(np.array(rgb, dtype=np.uint8)).permute(2, 0, 1).unsqueeze(0).float().div(255)
        x = x.to(self.device)
        v, e = self.cfg["vit"], self.cfg["effnet"]
        pooled = self.vit(pixel_values=self._normalise(x, v["size"], v["mean"], v["std"])).pooler_output
        return {"vit": float(self.head(pooled.float())), "effnet": float(self.eff(self._normalise(x, e["size"],
                                                                                               e["mean"], e["std"])))}


def card(name, logit, threshold, truth_fake, reveal):
    fake = logit > threshold
    colour = RED if fake else TEAL
    mark = (" ✓" if fake == truth_fake else " ✗") if reveal else ""
    return (f'<div style="flex:1;min-width:200px;padding:14px 18px;border-radius:12px;background:#ffffff;'
            f'border:2px solid {colour}">'
            f'<div style="font-size:15px;color:#333333">{name}</div>'
            f'<div style="font-size:32px;font-weight:700;line-height:1.25;color:{colour}">'
            f'{"FAKE" if fake else "GENUINE"}{mark}</div>'
            f'<div style="font-size:13px;color:#555555">score {logit:+.2f} · threshold {threshold:+.2f}</div></div>')


def build(folder):
    import gradio as gr

    det = Detectors(folder)
    cfg, faces = det.cfg, det.cfg["faces"]
    by_key = {f["key"]: f for f in faces}
    rgb = {f["key"]: np.asarray(Image.open(det.dir / "faces" / f["file"]).convert("RGB")) for f in faces}

    @lru_cache(maxsize=256)
    def score(key, quality):
        img = compress(rgb[key], QUALITIES[quality])
        return img, det.logits(img)

    for f in faces:  # warm up, so the first click in class is instant
        score(f["key"], "Original")

    def view(key, quality, reveal):
        img, lg = score(key, quality)
        f = by_key[key]
        cards = "".join(card(cfg[m]["name"], lg[m], cfg[m]["threshold"], f["fake"], reveal) for m in ("vit", "effnet"))
        answer = (f'<div style="margin-top:12px;font-size:17px"><b>Answer:</b> {f["truth"]}</div>' if reveal else "")
        html = f'<div style="display:flex;gap:12px;flex-wrap:wrap">{cards}</div>{answer}'
        return Image.fromarray(img), html

    with gr.Blocks(title="Real or fake?") as demo:
        gr.Markdown("## Real or fake?\nClick a face, then change the image quality. Each detector calls a face "
                    "**fake** when its score is above its fixed threshold (set so 1% of genuine FF++ faces are "
                    "rejected).")
        key = gr.State(faces[0]["key"])
        with gr.Row():
            with gr.Column(scale=3):
                gallery = gr.Gallery(value=[(Image.fromarray(rgb[f["key"]]), f["key"]) for f in faces],
                                     columns=4, rows=2, allow_preview=False, show_label=False, height="auto")
            with gr.Column(scale=2):
                shown = gr.Image(type="pil", show_label=False, interactive=False, height=300)
                quality = gr.Radio(list(QUALITIES), value="Original", label="Image quality")
                reveal = gr.Checkbox(value=False, label="Show the answer")
                verdict = gr.HTML()
        gr.Markdown(f"<sub>{cfg['credit']}</sub>")

        def pick(quality_value, reveal_value, evt: gr.SelectData):
            k = faces[evt.index]["key"]
            return (k, *view(k, quality_value, reveal_value))

        gallery.select(pick, [quality, reveal], [key, shown, verdict])
        quality.change(view, [key, quality, reveal], [shown, verdict])
        reveal.change(view, [key, quality, reveal], [shown, verdict])
        demo.load(view, [key, quality, reveal], [shown, verdict])
    return demo


def main():
    here = Path(__file__).resolve().parent
    default = here if (here / "config.json").is_file() else here / "outputs" / "demo"
    ap = argparse.ArgumentParser(description="Real or fake? In-class deepfake detection demo.")
    ap.add_argument("--dir", default=str(default), help="folder written by export_demo.py")
    ap.add_argument("--share", action="store_true", help="also create a public link")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    build(a.dir).launch(share=a.share, server_port=a.port, inbrowser=not a.share)


if __name__ == "__main__":
    main()
