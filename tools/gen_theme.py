"""Generate theme art for The Secret Garden with NetaYume (ComfyUI API), one image at a time, with a GPU temperature guard.

  python tools/gen_theme.py greek --kinds block            # all 12 blocks
  python tools/gen_theme.py greek --only block_3 mascot    # specific assets
  python tools/gen_theme.py greek --redo block_3           # overwrite
  python tools/gen_theme.py --list
Kinds: block, item, mascot, lock, fx, bg
Sticker assets are cut out with BiRefNet (ComfyUI), fx sprites are keyed from a black background.
"""
import argparse, io, json, os, shutil, subprocess, sys, time, urllib.parse, urllib.request, zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PIL import Image
import numpy as np
from theme_prompts import THEMES, STYLE, NEG, NEG_BG, FX_STYLE, FX_NEG, BG_STYLE

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
API = "http://127.0.0.1:8000"
INPUT = r"C:\Users\UsEr\Documents\ComfyUI\input"
OUTPUT = r"C:\Users\UsEr\Documents\ComfyUI\output"
TEMP_PAUSE, TEMP_RESUME, TEMP_ABORT, COOLDOWN = 78, 70, 86, 6


def gpu_temp():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        return int(float(r.stdout.strip().splitlines()[0]))
    except Exception:
        return 99  # unknown temperature counts as hot: never run blind


def guard():
    t = gpu_temp()
    if t >= TEMP_ABORT:
        print(f"GPU {t}C >= {TEMP_ABORT}C -> ABORT", flush=True); sys.exit(2)
    waited = 0
    while t >= TEMP_PAUSE and waited < 900:
        print(f"GPU {t}C hot, cooling...", flush=True); time.sleep(15); waited += 15; t = gpu_temp()
    return t


def post(p):
    r = urllib.request.urlopen(urllib.request.Request(API + "/prompt", json.dumps({"prompt": p}).encode(), {"Content-Type": "application/json"}))
    return json.load(r)["prompt_id"]


def wait(pid, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        h = json.load(urllib.request.urlopen(API + "/history/" + pid))
        if pid in h:
            return h[pid]
        time.sleep(1.5)
    raise TimeoutError(pid)


def fetch(h, node):
    im = h["outputs"][node]["images"][0]
    q = urllib.parse.urlencode({"filename": im["filename"], "subfolder": im["subfolder"], "type": im["type"]})
    data = urllib.request.urlopen(API + "/view?" + q).read()
    return Image.open(io.BytesIO(data))


def txt2img(prompt, neg, w, h, seed, prefix="secret-garden/misc"):
    p = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "NetaYume_v4_all_in_one.safetensors"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": neg, "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "5": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0],
                                                    "seed": seed, "steps": 30, "cfg": 4, "sampler_name": "euler", "scheduler": "normal", "denoise": 1}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": prefix}},
    }
    return fetch(wait(post(p)), "7").convert("RGB")


def cutout(img, prefix="secret-garden/misc"):
    name = "sg_cut_%d.png" % int(time.time() * 1000)
    img.save(os.path.join(INPUT, name))
    p = {
        "1": {"class_type": "LoadImage", "inputs": {"image": name}},
        "2": {"class_type": "LoadBackgroundRemovalModel", "inputs": {"bg_removal_name": "birefnet.safetensors"}},
        "3": {"class_type": "RemoveBackground", "inputs": {"bg_removal_model": ["2", 0], "image": ["1", 0]}},
        "4": {"class_type": "InvertMask", "inputs": {"mask": ["3", 0]}},
        "5": {"class_type": "JoinImageWithAlpha", "inputs": {"image": ["1", 0], "alpha": ["4", 0]}},
        "6": {"class_type": "SaveImage", "inputs": {"images": ["5", 0], "filename_prefix": prefix + "_cut"}},
    }
    out = fetch(wait(post(p)), "6").convert("RGBA")
    try:
        os.remove(os.path.join(INPUT, name))
    except OSError:
        pass
    return out


def fit_square(im, size, margin=.06):
    """Crop to the alpha bounding box, then centre it in a size x size canvas."""
    box = im.getchannel("A").point(lambda v: 255 if v > 24 else 0).getbbox()
    if box:
        im = im.crop(box)
    inner = int(size * (1 - 2 * margin)); k = inner / max(im.size)
    im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.LANCZOS)
    out = Image.new("RGBA", (size, size), (0, 0, 0, 0)); out.paste(im, ((size - im.width) // 2, (size - im.height) // 2), im)
    return out


def key_black(img, size=128):
    """Glowing sprite on pure black -> RGBA (alpha from brightness), so it blends well both normally and additively."""
    a = np.asarray(img.convert("RGB")).astype(np.float32) / 255
    m = a.max(axis=2)
    alpha = np.clip((m - .05) * 1.25, 0, 1)
    rgb = np.where(alpha[..., None] > 0.01, a / np.maximum(alpha[..., None], .05), 0).clip(0, 1)
    rgba = np.dstack([rgb, alpha]); im = Image.fromarray((rgba * 255).astype(np.uint8))
    return fit_square(im, size, .04)


def seed_of(theme, name):
    return zlib.crc32(f"{theme}/{name}".encode()) & 0x7FFFFFFF


def assets(theme):
    T = THEMES[theme]; out = {}
    for i, d in enumerate(T["blocks"], 1): out[f"block_{i}"] = ("block", d)
    for k, d in T["items"].items(): out[f"item_{k}"] = ("item", d)
    out["mascot"] = ("mascot", T["mascot"])
    out["lock"] = ("lock", T["lock"]); out["lock_hit"] = ("lock", T["lock_hit"])
    for i, d in enumerate(T["fx"], 1): out[f"fx_{i}"] = ("fx", d)
    for k in ("menu_land", "menu_port", "game_land", "game_port"): out[k] = ("bg", T[k])
    return out


def make(theme, name, kind, desc):
    seed = seed_of(theme, name) + int(os.environ.get("SEED_OFFSET", "0"))
    folder = os.path.join(ROOT, "themes", theme); os.makedirs(folder, exist_ok=True)
    T = THEMES[theme]
    pre = f"secret-garden/{theme}/{name}"  # unique name per asset, kept in ComfyUI/output/secret-garden/
    if kind in ("block", "item", "lock", "mascot"):
        prompt = f"{STYLE}, {desc}, {T['mood']}"
        img = txt2img(prompt, NEG, 768, 768, seed, pre)
        out = fit_square(cutout(img, pre), 384 if kind == "mascot" else 256, .05 if kind != "block" else .04)
        path = os.path.join(folder, name + ".png"); out.save(path, optimize=True)
    elif kind == "fx":
        img = txt2img(f"{FX_STYLE}, {desc}", FX_NEG, 640, 640, seed, pre)
        path = os.path.join(folder, name + ".png"); key_black(img).save(path, optimize=True)
    else:
        land = name.endswith("land")
        prompt = f"{BG_STYLE}, {desc}, {T['bg_mood']}"
        img = txt2img(prompt, NEG_BG, 1344 if land else 768, 768 if land else 1344, seed, pre)
        path = os.path.join(folder, name + ".jpg"); img.save(path, quality=90)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("theme", nargs="?")
    ap.add_argument("--kinds", default="")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--redo", nargs="*")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list or not a.theme:
        for t in THEMES: print(t, len(assets(t)), "assets")
        return
    todo = assets(a.theme)
    kinds = [k for k in a.kinds.split(",") if k]
    names = a.only or a.redo or [n for n, (k, _) in todo.items() if not kinds or k in kinds]
    force = bool(a.redo)
    ext = lambda k: ".jpg" if k == "bg" else ".png"
    n = 0
    for name in names:
        kind, desc = todo[name]
        path = os.path.join(ROOT, "themes", a.theme, name + ext(kind))
        if os.path.exists(path) and not force and not a.only:
            continue
        t = guard()
        t0 = time.time(); p = make(a.theme, name, kind, desc); n += 1
        print(f"[{n}] {a.theme}/{name} {time.time() - t0:.0f}s gpu {gpu_temp()}C (was {t}C)", flush=True)
        time.sleep(COOLDOWN)
    print("done", n, "images")


if __name__ == "__main__":
    main()
