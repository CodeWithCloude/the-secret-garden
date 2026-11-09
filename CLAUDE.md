# The Secret Garden

Single-file HTML merge game (`index.html`, canvas, 4x4 board) with per-theme art in `themes/<id>/`.

## Art rules (user, 2026-11-09)
- Style: ComfyUI + NetaYume, anime cozy everywhere: sticker blocks with dark brown outline, and backgrounds in flat anime cel shading with clean lines and soft pastel colors (never realistic or oil-painting looking; `BG_STYLE` / `NEG_BG` in `tools/theme_prompts.py`).
- Never use emoji in the game or in generated art/text.
- Never draw pictures with 2D code (canvas shapes) unless the user asks first. All art, sprites and particles come from ComfyUI; code only moves, rotates, glows and blends them.
- New themes also need: Bloom effect, sprite particles, and a difficulty set that fits the theme (see `THEMES[id].diff`).

## ComfyUI / GPU safety (RTX 4070 12GB)
- Generate with `tools/gen_theme.py` only: one image at a time, nvidia-smi temperature guard (pause 78C, abort 86C), cooldown between images.
- ComfyUI API is on `http://127.0.0.1:8000` (Comfy Desktop). Check it responds and check `nvidia-smi` (temp, VRAM) before a batch; never run two batches at once.
- Checkpoint `NetaYume_v4_all_in_one.safetensors`, euler / normal, 30 steps, cfg 4. Cutouts use BiRefNet through ComfyUI, never flood-fill.

## Generation workflow (user, 2026-11-09)
- Generate exactly one image per asset in a single pass (no multiple candidates, no regenerating while the pass runs). Review afterwards and redo only the bad ones (`--redo`).
- Every ComfyUI job gets a unique `filename_prefix` (`secret-garden/<theme>/<asset>`), and outputs are kept, never deleted. A shared prefix plus deleting outputs makes ComfyUI reuse file names, so its history shows the same picture over and over and looks like duplicate generation.

## Deploy
- Repo: `CodeWithCloude/the-secret-garden` (public, GitHub Pages from `main` root). Artifact "The Secret Garden" is updated after each release.
- Open the game with `#test` to unlock every theme and cycle them with the T key.
