"""
Main training script for the conditional DDPM on CIFAR-10.

Training uses reconstruction (noise-prediction) loss only.

Every epoch: generates a fixed-noise sample grid to visually track progress.
Every EVAL.FID_EVERY_N_EPOCHS epochs: computes an FID/IS estimate on a modest sample count.
At the end of training: computes a final, larger-sample-count FID/IS.
"""

import os

# Silence tqdm progress bars globally (e.g. any pretrained-weights download) --
# must be set before torch/torchvision are imported.
os.environ.setdefault("TQDM_DISABLE", "1")

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from PIL import Image, ImageDraw, ImageFont

from cifar10_dataset import CIFAR10Dataset, denormalize
from ConditionalDDPM import ConditionalDDPM
from diffusion_utils import GaussianDiffusion
from metrics import compute_fid_and_is


def load_config(path="configs.yml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_fixed_noise_inputs(cfg, device):
    """Fixed random noise + class ids cycling through 0..NUM_CLASSES-1, seeded, for
    consistent per-epoch qualitative samples. NUM_FIXED_SAMPLES can be set higher
    than NUM_CLASSES -- class ids simply wrap around (0,1,...,9,0,1,...,9,...) so
    every class keeps getting represented no matter how high it's set.
    """
    num_classes = cfg["MODEL"]["NUM_CLASSES"]
    num_samples = cfg["SAMPLING"]["NUM_FIXED_SAMPLES"]
    image_size = cfg["MODEL"]["IMAGE_SIDE_LENGTH"]

    generator = torch.Generator(device="cpu").manual_seed(cfg["TRAINING"]["SEED"])
    fixed_noise = torch.randn((num_samples, 3, image_size, image_size), generator=generator).to(device)
    fixed_class_ids = torch.arange(num_samples, device=device) % num_classes
    return fixed_noise, fixed_class_ids


@torch.no_grad()
def sample_with_fixed_noise(diffusion, model, fixed_noise, fixed_class_ids, device):
    """Runs the reverse diffusion loop starting from a fixed noise tensor (not resampled each call)."""
    model.eval()
    x_t = fixed_noise.clone()
    B = x_t.shape[0]

    for t_step in reversed(range(diffusion.timesteps)):
        t = torch.full((B,), t_step, device=device, dtype=torch.long)
        predicted_noise = model(x_t, fixed_class_ids, t.float())

        alpha_t = diffusion._extract(diffusion.alphas, t, x_t.shape)
        alpha_bar_t = diffusion._extract(diffusion.alpha_bars, t, x_t.shape)
        beta_t = diffusion._extract(diffusion.betas, t, x_t.shape)

        model_mean = (1.0 / torch.sqrt(alpha_t)) * (
            x_t - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * predicted_noise
        )

        if t_step > 0:
            posterior_var_t = diffusion._extract(diffusion.posterior_variance, t, x_t.shape)
            noise = torch.randn_like(x_t)
            x_t = model_mean + torch.sqrt(posterior_var_t) * noise
        else:
            x_t = model_mean

    model.train()
    return x_t.clamp(-1.0, 1.0)


CIFAR10_CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def save_labeled_sample_grid(images, class_ids, path, nrow, padding=2, upscale=8):
    """
    Saves an image grid (like torchvision.utils.save_image) but also stamps each
    tile's class name in its top-left corner, so it's easy to visually confirm
    samples match the class they were conditioned on.

    Args:
        images: (N, 3, H, W) tensor already denormalized to [0, 1].
        class_ids: (N,) tensor or list of int class indices, same order as images.
        path: output file path.
        nrow: images per row (same meaning as torchvision.utils.make_grid).
        padding: pixel padding between grid cells (matches make_grid's default of 2).
        upscale: integer factor to enlarge the grid before drawing text. CIFAR-10
            images are only 32x32, too small to fit legible text otherwise -- default
            of 8 brings each 32x32 tile up to 256x256.
    """
    grid = make_grid(images, nrow=nrow, padding=padding)
    grid_np = (grid.clamp(0.0, 1.0) * 255).byte().permute(1, 2, 0).cpu().numpy()
    grid_img = Image.fromarray(grid_np)

    if upscale > 1:
        grid_img = grid_img.resize(
            (grid_img.width * upscale, grid_img.height * upscale), resample=Image.NEAREST,
        )

    draw = ImageDraw.Draw(grid_img)
    font = ImageFont.load_default()

    img_h, img_w = images.shape[-2], images.shape[-1]
    ncols = nrow
    for idx in range(images.shape[0]):
        row, col = divmod(idx, ncols)
        cell_x = (padding + col * (img_w + padding)) * upscale
        cell_y = (padding + row * (img_h + padding)) * upscale
        class_id = int(class_ids[idx])
        label = CIFAR10_CLASS_NAMES[class_id] if 0 <= class_id < len(CIFAR10_CLASS_NAMES) else str(class_id)

        text_pos = (cell_x + 2, cell_y + 2)
        text_bbox = draw.textbbox(text_pos, label, font=font)
        box = (text_pos[0] - 1, text_pos[1] - 1, text_bbox[2] + 1, text_bbox[3] + 1)
        # small filled box sized to the text so it stays legible over any tile color
        draw.rectangle(box, fill=(0, 0, 0))
        draw.text(text_pos, label, fill=(255, 255, 0), font=font)

    grid_img.save(path)


def train_one_epoch(model, diffusion, train_loader, model_optimizer, cfg, device):
    """Runs the per-batch training loop for a single epoch (reconstruction loss only).

    No per-batch logging -- only returns the averaged loss so the caller can
    print a single summary line once the epoch finishes.

    Returns avg_recon.
    """
    grad_clip_norm = cfg["TRAINING"]["GRAD_CLIP_NORM"]

    running_recon = 0.0

    for x0, class_ids in train_loader:
        x0 = x0.to(device)
        class_ids = class_ids.to(device)
        B = x0.shape[0]

        t = torch.randint(0, diffusion.timesteps, (B,), device=device)
        noise = torch.randn_like(x0)
        x_t = diffusion.q_sample(x0, t, noise)

        predicted_noise = model(x_t, class_ids, t.float())

        recon_loss = nn.functional.mse_loss(predicted_noise, noise)

        model_optimizer.zero_grad()
        recon_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        model_optimizer.step()

        running_recon += recon_loss.item()

    n_batches = len(train_loader)
    avg_recon = running_recon / n_batches

    return avg_recon


def train(cfg):
    torch.manual_seed(cfg["TRAINING"]["SEED"])

    device = cfg["TRAINING"]["DEVICE"] if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    os.makedirs(cfg["TRAINING"]["CHECKPOINT_DIR"], exist_ok=True)
    os.makedirs(cfg["SAMPLING"]["SAMPLES_DIR"], exist_ok=True)

    model_cfg = cfg["MODEL"]
    embedding_dim = model_cfg["CLASS_EMBEDDING_DIM"]

    train_dataset = CIFAR10Dataset(
        root=cfg["DATA"]["DATA_DIR"], train=True,
        image_side_length=model_cfg["IMAGE_SIDE_LENGTH"], augment=True,
        download=cfg["DATA"].get("DOWNLOAD", True),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=cfg["DATA"]["BATCH_SIZE"], shuffle=True,
        num_workers=cfg["DATA"]["NUM_WORKERS"], pin_memory=True, drop_last=True,
    )

    real_eval_dataset = CIFAR10Dataset(
        root=cfg["DATA"]["DATA_DIR"], train=False,
        image_side_length=model_cfg["IMAGE_SIDE_LENGTH"], augment=False,
        download=cfg["DATA"].get("DOWNLOAD", True),
    )
    real_eval_loader = DataLoader(
        real_eval_dataset, batch_size=cfg["DATA"]["BATCH_SIZE"], shuffle=True,
        num_workers=cfg["DATA"]["NUM_WORKERS"],
    )

    # ---- Model / diffusion ----
    model = ConditionalDDPM(
        num_classes=model_cfg["NUM_CLASSES"],
        embedding_dim=embedding_dim,
        num_groups=model_cfg["NUM_GROUPS"],
        channels_per_level=model_cfg["CHANNELS_PER_LEVEL"],
        theta=model_cfg["THETA"],
    ).to(device)

    diffusion = GaussianDiffusion(
        timesteps=cfg["DIFFUSION"]["TIMESTEPS"],
        beta_start=cfg["DIFFUSION"]["BETA_START"],
        beta_end=cfg["DIFFUSION"]["BETA_END"],
        schedule=cfg["DIFFUSION"]["BETA_SCHEDULE"],
        device=device,
    )

    # ---- Optimizer ----
    betas = tuple(cfg["TRAINING"]["ADAM_BETAS"])
    model_optimizer = torch.optim.Adam(model.parameters(), lr=cfg["TRAINING"]["LR"], betas=betas)

    # ---- Fixed noise for per-epoch qualitative sampling ----
    fixed_noise, fixed_class_ids = build_fixed_noise_inputs(cfg, device)

    num_epochs = cfg["TRAINING"]["EPOCHS"]

    for epoch in range(1, num_epochs + 1):
        avg_recon = train_one_epoch(model, diffusion, train_loader, model_optimizer, cfg, device)

        print(f"== Epoch {epoch}/{num_epochs} done | avg_recon={avg_recon:.4f} ==")

        # ---- Per-epoch qualitative sample grid from fixed noise ----
        if epoch % cfg["SAMPLING"]["SAMPLE_EVERY_N_EPOCHS"] == 0:
            samples = sample_with_fixed_noise(diffusion, model, fixed_noise, fixed_class_ids, device)
            grid_path = os.path.join(cfg["SAMPLING"]["SAMPLES_DIR"], f"epoch_{epoch:03d}.png")
            save_labeled_sample_grid(
                denormalize(samples), fixed_class_ids, grid_path,
                nrow=min(model_cfg["NUM_CLASSES"], fixed_noise.shape[0]),
            )
            print(f"Saved fixed-noise sample grid -> {grid_path}")

        # ---- Periodic FID / IS estimate ----
        if cfg["EVAL"]["COMPUTE_FID_IS"] and epoch % cfg["EVAL"]["FID_EVERY_N_EPOCHS"] == 0:
            fid_value, is_mean, is_std = compute_fid_and_is(
                diffusion, model, real_eval_loader,
                num_samples=cfg["EVAL"]["FID_NUM_SAMPLES"],
                num_classes=model_cfg["NUM_CLASSES"], device=device,
            )
            print(f"[epoch {epoch}] periodic FID={fid_value:.3f} | IS={is_mean:.3f} +/- {is_std:.3f}")

        # ---- Checkpointing ----
        if epoch % cfg["TRAINING"]["CHECKPOINT_EVERY_N_EPOCHS"] == 0 or epoch == num_epochs:
            ckpt_path = os.path.join(cfg["TRAINING"]["CHECKPOINT_DIR"], f"ddpm_epoch_{epoch:03d}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "model_optimizer_state_dict": model_optimizer.state_dict(),
                "config": cfg,
            }, ckpt_path)
            print(f"Saved checkpoint -> {ckpt_path}")

    # ---- Final, larger-sample-count FID / IS ----
    if cfg["EVAL"]["COMPUTE_FID_IS"]:
        fid_value, is_mean, is_std = compute_fid_and_is(
            diffusion, model, real_eval_loader,
            num_samples=cfg["EVAL"]["FINAL_FID_NUM_SAMPLES"],
            num_classes=model_cfg["NUM_CLASSES"], device=device,
        )
        print(f"== FINAL METRICS == FID={fid_value:.3f} | IS={is_mean:.3f} +/- {is_std:.3f}")


def main():
    cfg = load_config("configs.yml")
    train(cfg)


if __name__ == "__main__":
    main()