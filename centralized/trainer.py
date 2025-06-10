import lightning as pl
import torch
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger

from datamodule import PostureDataModule
from model import PostureMLP

BATCH_SIZE = 32
NUM_WORKERS = 15

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    # Initialize model
    model = PostureMLP(learning_rate=0.001)

    # Setup logger
    logger = TensorBoardLogger(
        save_dir="logs", name="posture_mlp", version=None  # Auto-increment version
    )

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=3,
        filename="posture-{epoch:02d}-{val_acc:.2f}",
        save_last=True,
        verbose=True,
    )

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, mode="min", verbose=True
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Setup data module
    datamodule = PostureDataModule(
        csv_file="../datasets/short_train.csv",
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    # Enhanced trainer
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping, lr_monitor, RichProgressBar()],
        accelerator="auto",
        devices=1,
        min_epochs=10,
        max_epochs=100,
        precision="16-mixed",  # Mixed precision for faster training
        gradient_clip_val=1.0,  # Gradient clipping
        accumulate_grad_batches=1,
        log_every_n_steps=20,
        check_val_every_n_epoch=1,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=True,  # For reproducibility
    )

    # Training
    trainer.fit(model, datamodule)

    # Test with best checkpoint
    trainer.test(model, datamodule, ckpt_path="best")

    print(f"\nTraining completed!")
    print(f"TensorBoard logs saved to: {logger.log_dir}")
    print(f"To view results, run: tensorboard --logdir={logger.save_dir}")
