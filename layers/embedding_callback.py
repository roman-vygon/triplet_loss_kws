import time
import nemo
import torch

try:
    import wandb

    _WANDB_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    _WANDB_AVAILABLE = False

logging = nemo.logging
from nemo.core.callbacks import ActionCallback

import tensorflow as tf
import tensorboard as tb

tf.io.gfile = tb.compat.tensorflow_stub.io.gfile


class EmbeddingEvaluatorCallback(ActionCallback):
    """
    For callback documentation: please see
    https://nvidia.github.io/NeMo/tutorials/callbacks.html
    """

    def __init__(
            self,
            eval_tensors,
            user_iter_callback,
            user_epochs_done_callback,
            tb_writer=None,
            tb_writer_func=None,
            eval_step=1,
            eval_epoch=None,
            wandb_name=None,
            wandb_project=None,
            eval_at_start=True,
    ):
        # TODO: Eval_epoch currently does nothing
        if eval_step is None and eval_epoch is None:
            raise ValueError("Either eval_step or eval_epoch must be set. " f"But got: {eval_step} and {eval_epoch}")
        if (eval_step is not None and eval_step <= 0) or (eval_epoch is not None and eval_epoch <= 0):
            raise ValueError(f"Eval_step and eval_epoch must be > 0." f"But got: {eval_step} and {eval_epoch}")
        super().__init__()
        self._eval_tensors = eval_tensors
        self._swriter = tb_writer
        self._tb_writer_func = tb_writer_func
        self._eval_frequency = eval_step
        self._eval_at_start = eval_at_start
        # will be passed to callbacks below
        self._global_var_dict = {}

        # Callbacks
        self.user_iter_callback = user_iter_callback
        self.user_done_callback = user_epochs_done_callback

        # Weights and biases
        self._wandb_project = wandb_project
        self._wandb_name = wandb_name

    @property
    def eval_tensors(self):
        return self._eval_tensors

    @property
    def tb_writer_func(self):
        return self._tb_writer_func

    @property
    def swriter(self):
        return self._swriter

    def on_epoch_end(self):
        pass

    def on_iteration_end(self):
        if self.step == 0 and not self._eval_at_start:
            return
        if self.step % self._eval_frequency == 0:
            if self.global_rank == 0 or self.global_rank is None:
                logging.info('Doing Evaluation ' + '.' * 30)
            start_time = time.time()
            self.action._eval(self._eval_tensors, self, self.step)
            elapsed_time = time.time() - start_time
            if self.global_rank == 0 or self.global_rank is None:
                logging.info(f'Evaluation time: {elapsed_time} seconds')
            embeds = torch.cat(self._global_var_dict['embeds'], 0)[:, 0, :]
            labels = torch.cat(self._global_var_dict['labels'], 0)
            self._swriter.add_embedding(embeds, metadata=labels, global_step=self.step)

    def on_action_start(self):
        if self.global_rank is None or self.global_rank == 0:
            if self._wandb_name is not None or self._wandb_project is not None:
                if _WANDB_AVAILABLE and wandb.run is None:
                    wandb.init(name=self._wandb_name, project=self._wandb_project)
                elif _WANDB_AVAILABLE and wandb.run is not None:
                    logging.info("Re-using wandb session")
                else:
                    logging.error("Could not import wandb. Did you install it (pip install --upgrade wandb)?")
                    logging.info("Will not log data to weights and biases.")
                    self._wandb_name = None
                    self._wandb_project = None

    def on_action_end(self):
        step = self.step
        if self.global_rank == 0 or self.global_rank is None:
            logging.info('Final Evaluation ' + '.' * 30)
        start_time = time.time()
        self.action._eval(self._eval_tensors, self, step)
        elapsed_time = time.time() - start_time
        if self.global_rank == 0 or self.global_rank is None:
            logging.info(f'Evaluation time: {elapsed_time} seconds')

    def clear_global_var_dict(self):
        self._global_var_dict = {}

    def wandb_log(self, tensors_logged):
        if self._wandb_name is not None and _WANDB_AVAILABLE:
            wandb.log(tensors_logged, step=self.step)
