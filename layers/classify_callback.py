import os
import nemo

logging = nemo.logging
from nemo.core.callbacks import ActionCallback
import tensorflow as tf
import tensorboard as tb

tf.io.gfile = tb.compat.tensorflow_stub.io.gfile


class RunClassifierCallback(ActionCallback):

    def __init__(
            self,
            eval_step=1,
            eval_epoch=None,
            name='test',
            num_classes=35,
            gpu=1,
            hidden_size=64,
            manifest='100',
            model='Res8'
    ):
        if eval_step is None and eval_epoch is None:
            raise ValueError("Either eval_step or eval_epoch must be set. " f"But got: {eval_step} and {eval_epoch}")
        if (eval_step is not None and eval_step <= 0) or (eval_epoch is not None and eval_epoch <= 0):
            raise ValueError(f"Eval_step and eval_epoch must be > 0." f"But got: {eval_step} and {eval_epoch}")
        super().__init__()
        self._eval_frequency = eval_step
        self.num_classes = num_classes
        self.name = name
        self.gpu = gpu
        self.hidden_size = hidden_size
        self.manifest = manifest
        self.model = model

    def start_classification(self, step, manifest):
        os.system(
            f"python TripletClassifier.py --enc_name={self.name} --enc_step={step} --name={self.name + '_classifier' + str(step)} --num_classes={self.num_classes} --gpu={self.gpu} --hidden_size={self.hidden_size} --manifest={manifest} --model={self.model}")

    def on_iteration_end(self):
        if self.step == 0:
            return
        if self.step % self._eval_frequency == 0:
            if self.global_rank == 0 or self.global_rank is None:
                logging.info('Starting classification ' + '.' * 30)
            self.start_classification(self.step, self.manifest)

    def on_action_end(self):
        step = self.step
        if self.global_rank == 0 or self.global_rank is None:
            logging.info('Final Classification ' + '.' * 30)
        self.start_classification(step, self.manifest)
