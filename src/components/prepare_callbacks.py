import os
import time
from src.config.configuration import PrepareCallbacksConfig
import torch
from torch.utils.tensorboard import SummaryWriter


class PrepareCallback:
    def __init__(self, config:PrepareCallbacksConfig):
        self.config = config
        self.timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        self.tb_running_log_dir = os.path.join(
            self.config.tensorboard_root_log_dir,
            f"tb_logs_at_{self.timestamp}",
        )
        self._writer = None
        self.best_loss = float('inf')

    @property
    def _create_tb_callbacks(self):
        """Equivalent of TensorBoard callback in TensorFlow"""
        if self._writer is None:
            self._writer = SummaryWriter(log_dir=self.tb_running_log_dir)
        return self._writer

    @property
    def _create_ckpt_callbacks(self):
        """Equivalent of ModelCheckpoint in TensorFlow"""
        def checkpoint_callback(model, current_loss):
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                torch.save(model.state_dict(), self.config.checkpoint_model_filepath)
                print(f"[Checkpoint] Saved model with loss {current_loss:.4f}")
        return checkpoint_callback

    def get_tb_ckpt_callbacks(self):
        return [
            self._create_tb_callbacks,
            self._create_ckpt_callbacks
        ]

    def close(self):
        if self._writer is not None:
            self._writer.close()