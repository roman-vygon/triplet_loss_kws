import math
import os
from functools import partial
from ruamel.yaml import YAML
import nemo
import nemo.collections.asr as nemo_asr
from nemo.utils.lr_policies import CosineAnnealing
from nemo.collections.asr.helpers import (
    monitor_classification_training_progress,
    process_classification_evaluation_batch,
    process_classification_evaluation_epoch,
)
from models.resnet import Res15, Res8
import argparse

logging = nemo.logging
from models.classifier import ClassificationNet
from layers.l2 import L2Regularizer

import json
from models.fc import LinearLayer

parser = argparse.ArgumentParser(description='Triplet loss classifier')
parser.add_argument('--gpu', type=int, help='gpu#', default=0)
parser.add_argument('--name', type=str, help='logdir name', default='test')
parser.add_argument('--enc_name', type=str, help='name of encoder logdir name', default='test')
parser.add_argument('--enc_step', type=int, help='encoder checkpoint step', default=0)
parser.add_argument('--num_classes', type=int, help='number of classes', default=35)
parser.add_argument('--hidden_size', type=int, help='size of hidden layers', default=64)
parser.add_argument('--manifest', type=str, help='manifest', default='10')
parser.add_argument('--model', type=str, help='model', default='Res8')

args = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
manifests = json.load(open('manifests.json', 'r'))
data_dir = '.'

background_dataset = data_dir + '/google_dataset_v2/google_speech_recognition_v2/background_manifest.json'
train_dataset = manifests[args.manifest]['train']
val_dataset = manifests[args.manifest]['dev']

yaml = YAML(typ="safe")
with open(f"configs/words{args.manifest}.yaml") as f:
    jasper_params = yaml.load(f)

labels = jasper_params['labels']
sample_rate = jasper_params['sample_rate']

lr = 1e-3
num_epochs = 2
batch_size = 128
weight_decay = 0.001
num_classes = len(labels)

neural_factory = nemo.core.NeuralModuleFactory(
    log_dir=data_dir + '/runs/' + args.name,
    create_tb_writer=True)
tb_writer = neural_factory.tb_writer

train_data_layer = nemo_asr.AudioToSpeechLabelDataLayer(
    manifest_filepath=train_dataset,
    sample_rate=sample_rate,
    labels=labels,
    batch_size=batch_size,
    num_workers=0,
    shuffle=True,
)
eval_data_layer = nemo_asr.AudioToSpeechLabelDataLayer(
    manifest_filepath=val_dataset,
    sample_rate=sample_rate,
    labels=labels,
    batch_size=batch_size,
    num_workers=0,
    shuffle=True,
)

data_preprocessor = nemo_asr.AudioToMelSpectrogramPreprocessor(
    sample_rate=sample_rate, **jasper_params["AudioToMelSpectrogramPreprocessor"],
)
assert args.model in ['Res8', 'Res15', 'Quartz']

if args.model == 'Res8':
    encoder = Res8(args.hidden_size).to('cuda')
    encoder.restore_from(f'./runs/{args.enc_name}/checkpoints/Res8-STEP-{str(args.enc_step)}.pt')
elif args.model == 'Res15':
    encoder = Res15(args.hidden_size).to('cuda')
    encoder.restore_from(f'./runs/{args.enc_name}/checkpoints/Res15-STEP-{str(args.enc_step)}.pt')
elif args.model == 'Quartz':
    encoder = nemo_asr.JasperEncoder(**jasper_params["JasperEncoder"])
    fc = LinearLayer(64 * 256)  # TODO find shape from jasper_params
    encoder.restore_from(f'./runs/{args.enc_name}/checkpoints/JasperEncoder-STEP-{str(args.enc_step)}.pt')
    fc.restore_from(f'./runs/{args.enc_name}/checkpoints/LinearLayer-STEP-{str(args.enc_step)}.pt')
    fc.freeze()
encoder.freeze()

l2_regularizer = L2Regularizer()
decoder = ClassificationNet(num_classes, args.hidden_size).to('cuda')

N = len(train_data_layer)
steps_per_epoch = math.ceil(N / float(batch_size) + 1)

logging.info("Steps per epoch : {0}".format(steps_per_epoch))
logging.info('Have {0} examples to train on.'.format(N))

ce_loss = nemo_asr.CrossEntropyLossNM()

logging.info('================================')
logging.info(f"Number of parameters in encoder: {encoder.num_weights}")
logging.info(f"Number of parameters in decoder: {decoder.num_weights}")
logging.info(
    f"Total number of parameters in model: " f"{decoder.num_weights + encoder.num_weights}"
)
logging.info('================================')

"""BUILDING TRAIN GRAPH"""

audio_signal, audio_signal_len, commands, command_len = train_data_layer()
processed_signal, processed_signal_len = data_preprocessor(input_signal=audio_signal, length=audio_signal_len)

encoded, encoded_len = encoder(audio_signal=processed_signal, length=processed_signal_len)

if args.model == 'Quartz':
    encoded = fc(embeddings=encoded)

encoded = l2_regularizer(embeds=encoded)
decoded = decoder(embeddings=encoded)

train_loss = ce_loss(logits=decoded, labels=commands)

"""BUILDING TEST GRAPH"""

test_audio_signal, test_audio_signal_len, test_commands, test_command_len = eval_data_layer()
test_processed_signal, test_processed_signal_len = data_preprocessor(
    input_signal=test_audio_signal, length=test_audio_signal_len
)

test_encoded, test_encoded_len = encoder(audio_signal=test_processed_signal, length=test_processed_signal_len)
if args.model == 'Quartz':
    test_encoded = fc(embeddings=test_encoded)

test_encoded = l2_regularizer(embeds=test_encoded)
test_decoded = decoder(embeddings=test_encoded)

test_loss = ce_loss(logits=test_decoded, labels=test_commands)

train_callback = nemo.core.SimpleLossLoggerCallback(
    tensors=[train_loss, decoded, commands],
    print_func=partial(monitor_classification_training_progress, eval_metric=None),
    get_tb_values=lambda x: [("loss", x[0])],
    tb_writer=neural_factory.tb_writer,
)

tagname = 'TestSet'
eval_callback = nemo.core.EvaluatorCallback(
    eval_tensors=[test_loss, test_decoded, test_commands],
    user_iter_callback=partial(process_classification_evaluation_batch, top_k=1),
    user_epochs_done_callback=partial(process_classification_evaluation_epoch, eval_metric=1, tag=tagname),
    eval_step=500,
    tb_writer=neural_factory.tb_writer,
    eval_at_start=False
)

chpt_callback = nemo.core.CheckpointCallback(
    folder=neural_factory.checkpoint_dir,
    step_freq=500,
    checkpoints_to_keep=100
)

callbacks = [train_callback, eval_callback, chpt_callback]

lr_policy = CosineAnnealing(
    total_steps=num_epochs * steps_per_epoch,
    warmup_ratio=0.05,
    min_lr=1e-4,
)

logging.info(f"Using `{lr_policy}` Learning Rate Scheduler")

neural_factory.train(
    tensors_to_optimize=[train_loss],
    callbacks=callbacks,
    lr_policy=lr_policy,
    optimizer="novograd",
    optimization_params={
        "num_epochs": num_epochs,
        "max_steps": None,
        "lr": lr,
        "momentum": 0.95,
        "betas": (0.98, 0.5),
        "weight_decay": weight_decay,
        "grad_norm_clip": None,
    },
    batches_per_step=1,
)
