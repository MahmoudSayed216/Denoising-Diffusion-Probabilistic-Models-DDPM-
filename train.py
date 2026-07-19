"""
Main training script for the conditional DDPM on CIFAR-10.

Progressive loss schedule (see configs.yml -> LOSS):
    Phase 1 (epochs 1..PHASE1_END_EPOCH):                reconstruction only
    Phase 2 (epochs PHASE1_END_EPOCH+1..PHASE2_END_EPOCH): + perceptual (VGG16)
    Phase 3 (epochs PHASE2_END_EPOCH+1..EPOCHS):           + adversarial (PatchGAN)

Every epoch: generates a fixed-noise sample grid to visually track progress.
Every EVAL.FID_EVERY_N_EPOCHS epochs: computes an FID/IS estimate on a modest sample count.
At the end of training: computes a final, larger-sample-count FID/IS.
"""

import os

# Silence tqdm progress bars globally (e.g. the VGG16 pretrained-weights
# download inside VGGPerceptualLoss) -- must be set before torch/torchvision
# are imported.
os.environ.setdefault("TQDM_DISABLE", "1")

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from cifar10_dataset import CIFAR10Dataset, denormalize
from ConditionalDDPM import ConditionalDDPM
from diffusion_utils import GaussianDiffusion
from losses import VGGPerceptualLoss, PatchDiscriminator, discriminator_loss, generator_adversarial_loss
from metrics import compute_fid_and_is


def load_config(path="configs.yml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_loss_weights(epoch, loss_cfg):
    """Returns (perceptual_weight, adversarial_weight) active at this epoch (1-indexed)."""
    perceptual_weight = 0.0
    adversarial_weight = 0.0

    if epoch > loss_cfg["PHASE1_END_EPOCH"]:
        perceptual_weight = loss_cfg["PERCEPTUAL_WEIGHT"]
    if epoch > loss_cfg["PHASE2_END_EPOCH"]:
        adversarial_weight = loss_cfg["ADVERSARIAL_WEIGHT"]

    return perceptual_weight, adversarial_weight


def build_fixed_noise_inputs(cfg, device):
    """Fixed random noise + one class per label, seeded, for consistent per-epoch qualitative samples."""
    num_classes = cfg["MODEL"]["NUM_CLASSES"]
    num_samples = min(cfg["SAMPLING"]["NUM_FIXED_SAMPLES"], num_classes)
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


def train_one_epoch(
    model, diffusion, discriminator, perceptual_loss_fn,
    train_loader, model_optimizer, disc_optimizer,
    epoch, cfg, device,
):
    """Runs the per-batch training loop for a single epoch.

    No per-batch logging -- only returns the averaged losses so the caller
    can print a single summary line once the epoch finishes.

    Returns (avg_recon, avg_perceptual, avg_adv, phase_desc).
    """
    perceptual_weight, adversarial_weight = get_loss_weights(epoch, cfg["LOSS"])
    phase_desc = "recon"
    if perceptual_weight > 0:
        phase_desc += "+perceptual"
    if adversarial_weight > 0:
        phase_desc += "+adversarial"

    grad_clip_norm = cfg["TRAINING"]["GRAD_CLIP_NORM"]

    running_recon = 0.0
    running_perceptual = 0.0
    running_adv = 0.0

    for x0, class_ids in train_loader:
        x0 = x0.to(device)
        class_ids = class_ids.to(device)
        B = x0.shape[0]

        t = torch.randint(0, diffusion.timesteps, (B,), device=device)
        noise = torch.randn_like(x0)
        x_t = diffusion.q_sample(x0, t, noise)

        predicted_noise = model(x_t, class_ids, t.float())

        # ---- Reconstruction loss (always active) ----
        recon_loss = nn.functional.mse_loss(predicted_noise, noise)
        total_loss = recon_loss
        perceptual_loss_value = torch.tensor(0.0, device=device)
        adv_gen_loss_value = torch.tensor(0.0, device=device)

        need_x0_hat = perceptual_weight > 0 or adversarial_weight > 0
        if need_x0_hat:
            x0_hat = diffusion.predict_x0_from_noise(x_t, t, predicted_noise)

        # ---- Perceptual loss (phase 2+) ----
        if perceptual_weight > 0:
            perceptual_loss_value = perceptual_loss_fn(x0_hat, x0)
            total_loss = total_loss + perceptual_weight * perceptual_loss_value

        # ---- Adversarial loss (phase 3+) ----
        if adversarial_weight > 0:
            # 1) Discriminator update (uses a detached fake so gradients don't flow into the model)
            disc_optimizer.zero_grad()
            d_loss = discriminator_loss(discriminator, x0, x0_hat.detach())
            d_loss.backward()
            disc_optimizer.step()

            # 2) Generator (model) adversarial term -- fresh forward through D, not detached
            adv_gen_loss_value = generator_adversarial_loss(discriminator, x0_hat)
            total_loss = total_loss + adversarial_weight * adv_gen_loss_value

        model_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        model_optimizer.step()

        running_recon += recon_loss.item()
        running_perceptual += perceptual_loss_value.item()
        running_adv += adv_gen_loss_value.item()

    n_batches = len(train_loader)
    avg_recon = running_recon / n_batches
    avg_perceptual = running_perceptual / n_batches
    avg_adv = running_adv / n_batches

    return avg_recon, avg_perceptual, avg_adv, phase_desc


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
        image_side_length=model_cfg["IMAGE_SIDE_LENGTH"], augment=True, download=True,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=cfg["DATA"]["BATCH_SIZE"], shuffle=True,
        num_workers=cfg["DATA"]["NUM_WORKERS"], pin_memory=True, drop_last=True,
    )

    real_eval_dataset = CIFAR10Dataset(
        root=cfg["DATA"]["DATA_DIR"], train=False,
        image_side_length=model_cfg["IMAGE_SIDE_LENGTH"], augment=False, download=True,
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

    # ---- Progressive loss components (perceptual / adversarial) ----
    perceptual_loss_fn = VGGPerceptualLoss(layer_indices=cfg["LOSS"]["VGG_LAYER_INDICES"]).to(device)
    discriminator = PatchDiscriminator().to(device)

    # ---- Optimizers ----
    betas = tuple(cfg["TRAINING"]["ADAM_BETAS"])
    model_optimizer = torch.optim.Adam(model.parameters(), lr=cfg["TRAINING"]["LR"], betas=betas)
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=cfg["TRAINING"]["DISCRIMINATOR_LR"], betas=betas)

    # ---- Fixed noise for per-epoch qualitative sampling ----
    fixed_noise, fixed_class_ids = build_fixed_noise_inputs(cfg, device)

    num_epochs = cfg["TRAINING"]["EPOCHS"]

    for epoch in range(1, num_epochs + 1):
        avg_recon, avg_perceptual, avg_adv, phase_desc = train_one_epoch(
            model, diffusion, discriminator, perceptual_loss_fn,
            train_loader, model_optimizer, disc_optimizer,
            epoch, cfg, device,
        )

        print(
            f"== Epoch {epoch}/{num_epochs} done | phase={phase_desc} | "
            f"avg_recon={avg_recon:.4f} "
            f"avg_perceptual={avg_perceptual:.4f} "
            f"avg_adv={avg_adv:.4f} =="
        )

        # ---- Per-epoch qualitative sample grid from fixed noise ----
        if epoch % cfg["SAMPLING"]["SAMPLE_EVERY_N_EPOCHS"] == 0:
            samples = sample_with_fixed_noise(diffusion, model, fixed_noise, fixed_class_ids, device)
            grid_path = os.path.join(cfg["SAMPLING"]["SAMPLES_DIR"], f"epoch_{epoch:03d}.png")
            save_image(denormalize(samples), grid_path, nrow=fixed_noise.shape[0])
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
                "discriminator_state_dict": discriminator.state_dict(),
                "model_optimizer_state_dict": model_optimizer.state_dict(),
                "disc_optimizer_state_dict": disc_optimizer.state_dict(),
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