import os
import math
import json

import nemo
import nemo.collections.asr as nemo_asr
from nemo.utils.lr_policies import CosineAnnealing
from layers.helper import (monitor_triplet_encoder_training_progress,
                           process_encoder_evaluation_batch,
                           process_encoder_evaluation_epoch)

from functools import partial
from ruamel.yaml import YAML

from loss.triplet import OnlineTripletLoss
from loss.utils import RandomNegativeTripletSelector
from layers.datalayer import BalancedAudioToSpeechLabelDataLayer

from models.resnet import Res15, Res8
import argparse

import numpy as np

logging = nemo.logging
from layers.l2 import L2Regularizer

from layers.embedding_callback import EmbeddingEvaluatorCallback
from layers.classify_callback import RunClassifierCallback

from models.fc import LinearLayer

data_dir = '.'

parser = argparse.ArgumentParser(description='Triplet loss encoder')

parser.add_argument('--lr', type=float, help='initial learning rate', default=1e-3)
parser.add_argument('--lr_end', type=float, help='final learning rate', default=8e-5)

parser.add_argument('--batch_classes', type=int, help='how many classes should be in a single batch', default=35)
parser.add_argument('--per_class', type=int, help='how many objects of each class should be in a single batch',
                    default=10)

parser.add_argument('--num_epochs', type=int, help='number of epochs', default=15)
parser.add_argument('--margin', type=float, help='margin for triplet loss', default=0.5)

parser.add_argument('--hidden_size', type=int, help='number of feature maps for resnet model', default=45)
parser.add_argument('--augment', dest='augment', help='whether to use augmentation', action='store_true', default=False)

parser.add_argument('--save_emb', dest='save_emb', help='whether to save validation set embeddings for tf projector',
                    action='store_true', default=False)

parser.add_argument('--cl', dest='cl', help='whether to run classification phase', action='store_true', default=False)

parser.add_argument('--gpu', type=int, help='gpu#', default=0)
parser.add_argument('--classify_gpu', type=int, help='which gpu to use for classification', default=1)

parser.add_argument('--manifest', type=str,
                    help='manifest number, 10-10000 for LibriWords, 36 for GoogleSpeechCommands', default='100')
parser.add_argument('--model', type=str, help='encoder architecture, can be Res8, Res15, Quartz', default='Res8')

parser.add_argument('--data_probs', type=int, help='sampling method (0-6)', default=0)

parser.add_argument('--name', type=str, help='logdir name', default='test')

args = parser.parse_args()

manifests = json.load(open('manifests.json', 'r'))
background_dataset = data_dir + '/google_dataset_v2/google_speech_recognition_v2/background_manifest.json'

train_dataset = manifests[args.manifest]['train']
val_dataset = manifests[args.manifest]['dev']

yaml = YAML(typ="safe")

config_name = f'words{args.manifest}.yaml'

with open("configs/" + config_name) as f:
    jasper_params = yaml.load(f)

labels = jasper_params['labels']
sample_rate = jasper_params['sample_rate']

dists = np.load('dists.npy')
man_number = int(args.manifest)

probs = np.load(f'files/class_probs{man_number}.npy')

if args.manifest == '10000':
    man_number = 9998

dists = dists[:man_number]
dists = dists[np.where(dists < man_number)]
dists = dists.reshape((man_number, man_number))

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
lr = args.lr
num_epochs = args.num_epochs
batch_size = args.batch_classes * args.per_class
weight_decay = 0.001
num_classes = len(labels)

neural_factory = nemo.core.NeuralModuleFactory(
    log_dir=data_dir + '/runs/' + args.name,
    create_tb_writer=True)
tb_writer = neural_factory.tb_writer

audio_augmentor = None

if args.augment:
    audio_augmentor = jasper_params.get('AudioAugmentor', None)
    audio_augmentor['noise']['manifest_path'] = background_dataset

train_data_layer = BalancedAudioToSpeechLabelDataLayer(
    manifest_filepath=train_dataset,
    labels=labels,
    sample_rate=sample_rate,
    batch_size=args.batch_classes * args.per_class,
    num_workers=0,
    augmentor=audio_augmentor,
    shuffle=True,
    num_classes=args.batch_classes,
    class_dists=dists,
    class_probs=probs,
    probs_num=args.data_probs
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

N = len(train_data_layer)
steps_per_epoch = math.ceil(N / float(batch_size) + 1)

logging.info("Steps per epoch : {0}".format(steps_per_epoch))
logging.info('Have {0} examples to train on.'.format(N))

spectr_augment_config = jasper_params.get('SpectrogramAugmentation', None)

if spectr_augment_config:
    data_spectr_augmentation = nemo_asr.SpectrogramAugmentation(**spectr_augment_config)

l2_regularizer = L2Regularizer()

assert args.model in ['Res8', 'Res15', 'Quartz']

if args.model == 'Res8':
    encoder = Res8(args.hidden_size).to('cuda')
elif args.model == 'Res15':
    encoder = Res15(args.hidden_size).to('cuda')
elif args.model == 'Quartz':
    encoder = nemo_asr.JasperEncoder(**jasper_params["JasperEncoder"])
    fc = LinearLayer(64 * 256)  # TODO find shape from jasper_params

triplet_loss = OnlineTripletLoss(args.margin, RandomNegativeTripletSelector(args.margin))

logging.info('================================')
logging.info(f"Number of parameters in encoder: {encoder.num_weights}")
logging.info('================================')

audio_signal, audio_signal_len, commands, command_len = train_data_layer()
processed_signal, processed_signal_len = data_preprocessor(input_signal=audio_signal, length=audio_signal_len)

if spectr_augment_config:
    processed_signal = data_spectr_augmentation(input_spec=processed_signal)

encoded, encoded_len = encoder(audio_signal=processed_signal, length=processed_signal_len)
if args.model == 'Quartz':
    encoded = fc(encoder_output=encoded)
encoded = l2_regularizer(embeds=encoded)
train_loss = triplet_loss(embeds=encoded, targets=commands)

test_audio_signal, test_audio_signal_len, test_commands, test_command_len = eval_data_layer()
test_processed_signal, test_processed_signal_len = data_preprocessor(
    input_signal=test_audio_signal, length=test_audio_signal_len
)

test_encoded, test_encoded_len = encoder(audio_signal=test_processed_signal, length=test_processed_signal_len)
if args.model == 'Quartz':
    test_encoded = fc(encoder_output=test_encoded)
test_encoded = l2_regularizer(embeds=test_encoded)
test_loss = triplet_loss(embeds=test_encoded, targets=test_commands)

"""SETUP CALLBACKS"""
train_callback = nemo.core.SimpleLossLoggerCallback(
    tensors=[train_loss],
    print_func=partial(monitor_triplet_encoder_training_progress, eval_metric=None),
    get_tb_values=lambda x: [("loss", x[0])],
    tb_writer=neural_factory.tb_writer,
)

chpt_callback = nemo.core.CheckpointCallback(
    folder=neural_factory.checkpoint_dir,
    step_freq=100,
    checkpoints_to_keep=100
)

callbacks = [train_callback, chpt_callback]

if args.save_emb:
    eval_callback = EmbeddingEvaluatorCallback(
        eval_tensors=[test_loss, test_commands, test_encoded],
        user_iter_callback=partial(process_encoder_evaluation_batch),
        user_epochs_done_callback=partial(process_encoder_evaluation_epoch, tag='TestSet'),
        eval_step=100,
        tb_writer=neural_factory.tb_writer,
        eval_at_start=False
    )
    callbacks.append(eval_callback)

if args.cl:
    classify_callback = RunClassifierCallback(
        eval_step=100,
        name=args.name,
        num_classes=len(labels),
        gpu=args.classify_gpu,
        hidden_size=args.hidden_size,
        manifest=args.manifest,
        model=args.model
    )
    callbacks.append(classify_callback)

lr_policy = CosineAnnealing(
    total_steps=num_epochs * steps_per_epoch,
    warmup_ratio=0.05,
    min_lr=args.lr_end,
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
