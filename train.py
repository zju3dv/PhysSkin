"""Train the PhysSkin point-shape-handle model on RigNet."""

import argparse
import logging
import math
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import pytorch_lightning as pl
import torch
import torch.multiprocessing as mp
from conflictfree.grad_operator import ConFIG_update
from conflictfree.utils import apply_gradient_vector, get_gradient_vector
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from physskin.data import RigNetDataset
from physskin.losses import compute_losses_batched_obj_loop
from physskin.model import PhysSkinPointShapeHandleModel


logging.basicConfig(level=logging.INFO, stream=sys.stdout)
LOGGER = logging.getLogger(__name__)
mp.set_start_method("spawn", force=True)


class SavePTCallback(Callback):
    """Save a compact model checkpoint in ``.pt`` format."""

    def __init__(self, checkpoint_dir, save_every=1, config=None):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.save_every = save_every
        self.config = config

    def on_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if (epoch + 1) % self.save_every != 0:
            return
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        save_path = os.path.join(
            self.checkpoint_dir, f"model_epoch_{epoch + 1}.pt"
        )
        torch.save(
            {
                "model_state": pl_module.model.state_dict(),
                "config": self.config,
            },
            save_path,
        )
        print(f"\nSaved model at epoch {epoch + 1} -> {save_path}")


class PhysSkinLightningModule(pl.LightningModule):
    def __init__(self, config, dataloader):
        super().__init__()
        self.config = config
        self.dataloader = dataloader
        self.model = PhysSkinPointShapeHandleModel(
            spatial_dimensions=3,
            layer_width=config["layer_width"],
            num_handles=config["num_handles"],
            num_layers=config["model_layers"],
            multires=config["multires"],
            num_self_attn=config["num_self_attn"],
            num_handle_tokens=config["num_handle_tokens"],
        )
        self.youngs_modulus = config["yms"]
        self.poisson_ratio = config["prs"]
        self.density = config["rhos"]
        self.approximate_volume = torch.tensor(
            [config["appx_vol"]], dtype=torch.float32
        )
        self.total_objects = len(dataloader)
        self.total_iterations = config["num_epochs"] * self.total_objects
        self.automatic_optimization = False

    def forward(self, latents, sampled_points, energy_interpolation):
        self.model.set_latent_feature(latents)
        return compute_losses_batched_obj_loop(
            model=self.model,
            normalized_pts=sampled_points,
            yms=self.youngs_modulus,
            prs=self.poisson_ratio,
            rhos=self.density,
            en_interp=energy_interpolation,
            batch_size=self.config["training_batch_size"],
            num_handles=self.config["num_handles"],
            appx_vol=self.approximate_volume,
            le_coeff=self.config["training_le_coeff"],
            lo_coeff=self.config["training_lo_coeff"],
        )

    def training_step(self, batch, batch_index):
        del batch_index
        latents_cpu, sampled_points_cpu, _ = batch
        latents = latents_cpu.float().to(self.device).squeeze(1)
        sampled_points = sampled_points_cpu.float().to(self.device)
        epoch = self.current_epoch
        energy_interpolation = float(epoch / self.config["num_epochs"])
        elastic_loss, orthogonality_loss = self(
            latents, sampled_points, energy_interpolation
        )

        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()
        gradients = []
        optimizer.zero_grad()
        self.manual_backward(elastic_loss, retain_graph=True)
        gradients.append(get_gradient_vector(self.model))
        optimizer.zero_grad()
        self.manual_backward(orthogonality_loss, retain_graph=True)
        gradients.append(get_gradient_vector(self.model))
        apply_gradient_vector(self.model, ConFIG_update(gradients))
        optimizer.step()
        scheduler.step()

        total_loss = elastic_loss + orthogonality_loss
        current_lr = scheduler.get_last_lr()[0]
        self.log("Loss/le", elastic_loss, prog_bar=True, on_step=True)
        self.log("Loss/lo", orthogonality_loss, prog_bar=True, on_step=True)
        self.log("Loss/total_loss", total_loss, prog_bar=True, on_step=True)
        self.log("lr/current", current_lr, prog_bar=True, on_step=True)
        self.log(
            "lr/en_interp", energy_interpolation, prog_bar=True, on_step=True
        )
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config["training_lr_start"]
        )
        warmup_iterations = int(self.total_iterations * 0.01)
        minimum_lr = self.config["training_lr_end"]
        base_lr = self.config["training_lr_start"]

        def learning_rate_multiplier(current_step):
            if current_step < warmup_iterations:
                return float(current_step) / float(max(1, warmup_iterations))
            progress = (
                (current_step - warmup_iterations)
                / (self.total_iterations - warmup_iterations)
            )
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return (
                (minimum_lr / base_lr)
                + (1 - minimum_lr / base_lr) * cosine_decay
            )

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, learning_rate_multiplier
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def train_dataloader(self):
        return self.dataloader

    def lr_scheduler_step(self, scheduler, optimizer_idx: int, metric=None):
        del optimizer_idx, metric
        scheduler.step()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument(
        "--split_dir",
        default=os.path.join(os.path.dirname(__file__), "data", "splits"),
    )
    parser.add_argument("--split", choices=["train", "val", "test"],
                        default="train")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--training_lr_start", type=float, default=5e-4)
    parser.add_argument("--training_lr_end", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--batch_obj", type=int, default=24)
    parser.add_argument("--batch_transform", type=int, default=32)
    parser.add_argument("--log_path", default="./logs")
    parser.add_argument("--ckpt_path", default="./checkpoints")
    parser.add_argument("--num_self_attn", type=int, default=8)
    parser.add_argument("--handles_num", type=int, default=16)
    parser.add_argument("--num_handle_tokens", type=int, default=None)
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=[0])
    parser.add_argument("--save_every", type=int, default=10)
    return parser.parse_args()


def make_training_config(args):
    return {
        "yms": 5e5,
        "prs": 0.45,
        "rhos": 500,
        "appx_vol": 0.5,
        "num_handles": args.handles_num,
        "model_layers": 2,
        "layer_width": 512,
        "multires": 10,
        "training_num_steps": 30000,
        "training_lr_start": args.training_lr_start,
        "training_lr_end": args.training_lr_end,
        "training_batch_size": args.batch_transform,
        "batch_obj": args.batch_obj,
        "category_ids": {"00000001": 55},
        "split": args.split,
        "ckpt_path": args.ckpt_path,
        "num_epochs": args.num_epochs,
        "log_path": args.log_path,
        "training_le_coeff": 1e-1,
        "training_lo_coeff": 1e6,
        "num_self_attn": args.num_self_attn,
        "num_handle_tokens": (
            args.handles_num
            if args.num_handle_tokens is None
            else args.num_handle_tokens
        ),
        "model_type": "PhysSkin_PointShapeHandle_CrossOnly_Model",
    }


def main():
    args = parse_args()
    config = make_training_config(args)
    os.makedirs(args.log_path, exist_ok=True)
    os.makedirs(args.ckpt_path, exist_ok=True)
    LOGGER.info("Training PhysSkin point-shape-handle model on RigNet")

    dataset = RigNetDataset(
        args.dataset_root,
        split=args.split,
        num_points=args.num_samples,
        split_dir=args.split_dir,
    )
    if not dataset:
        raise RuntimeError(
            f"No RigNet samples found in {args.dataset_root!r} for "
            f"split {args.split!r}"
        )
    train_loader = DataLoader(
        dataset,
        batch_size=config["batch_obj"],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    if not train_loader:
        raise RuntimeError(
            "The dataloader is empty; reduce --batch_obj or add more samples."
        )

    module = PhysSkinLightningModule(config, train_loader)
    tensorboard_logger = TensorBoardLogger(
        save_dir=args.log_path, name="PhysSkin"
    )
    save_callback = SavePTCallback(
        args.ckpt_path, save_every=args.save_every, config=config
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.gpu_ids,
        strategy="ddp" if len(args.gpu_ids) > 1 else None,
        max_epochs=args.num_epochs,
        logger=tensorboard_logger,
        callbacks=[save_callback],
    )
    trainer.fit(module)


if __name__ == "__main__":
    main()
