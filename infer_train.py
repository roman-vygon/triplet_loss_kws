import argparse
import math
import os

import nemo
import nemo.collections.asr as nemo_asr
import numpy as np
from ruamel.yaml import YAML

from models.resnet import Res8, Res15

logging = nemo.logging
from layers.l2 import L2Regularizer
import json
from models.fc import LinearLayer
from sklearn.metrics import f1_score
import faiss


class FaissKNeighbors:
    def __init__(self, k=5):
        self.index = None
        self.y = None
        self.k = k

    def fit(self, X, y):
        self.index = faiss.IndexFlatIP(X.shape[1])
        self.index.add(X.astype(np.float32))
        self.y = y

    def predict(self, X):
        distances, indices = self.index.search(X.astype(np.float32), k=self.k)
        votes = self.y[indices]
        predictions = np.array([np.argmax(np.bincount(x)) for x in votes])
        return predictions


parser = argparse.ArgumentParser(description='Model evaluation')
parser.add_argument('--gpu', type=int, help='gpu#', default=0)
parser.add_argument('--name', type=str, help='logdir name', default='test')
parser.add_argument('--enc_step', type=int, help='encoder checkpoint step', default=0)
parser.add_argument('--manifest', type=str, help='number of classes', default='10')
parser.add_argument('--hidden_size', type=int, help='size of hidden layers', default=64)
parser.add_argument('--model', type=str, help='encoder architecture, can be Res8, Res15, Quartz', default='Res8')
parser.add_argument('--k', type=int, help='kneighbours', default=5)
parser.add_argument('--save', dest='save', help='save embeddings and targets', action='store_true', default=False)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

data_dir = '.'
manifests = json.load(open('manifests.json', 'r'))
train_dataset = manifests[args.manifest]['train']
test_dataset = manifests[args.manifest]['test']

yaml = YAML(typ="safe")
with open(f"configs/words{args.manifest}.yaml") as f:
    jasper_params = yaml.load(f)

labels = jasper_params['labels']
tmp_labels = labels
sample_rate = jasper_params['sample_rate']

batch_size = 128
num_classes = len(labels)

logdir = data_dir + '/runs/' + args.name

neural_factory = nemo.core.NeuralModuleFactory(
    log_dir=logdir,
    create_tb_writer=True)
tb_writer = neural_factory.tb_writer

train_data_layer = nemo_asr.AudioToSpeechLabelDataLayer(
    manifest_filepath=train_dataset,
    labels=labels,
    sample_rate=sample_rate,
    batch_size=batch_size,
    num_workers=0,
    augmentor=None,
    shuffle=True
)
eval_data_layer = nemo_asr.AudioToSpeechLabelDataLayer(
    manifest_filepath=test_dataset,
    sample_rate=sample_rate,
    labels=labels,
    batch_size=batch_size,
    num_workers=0,
    shuffle=True,
)

data_preprocessor = nemo_asr.AudioToMelSpectrogramPreprocessor(
    sample_rate=sample_rate, **jasper_params["AudioToMelSpectrogramPreprocessor"],
)

if args.model == 'Res8':
    encoder = Res8(args.hidden_size).to('cuda')
    encoder.restore_from('./runs/{0}/checkpoints/Res8-STEP-{1}.pt'.format(args.name, str(args.enc_step)))
    encoder.freeze()
elif args.model == 'Res15':
    encoder = Res15(args.hidden_size).to('cuda')
    encoder.restore_from('./runs/{0}/checkpoints/Res15-STEP-{1}.pt'.format(args.name, str(args.enc_step)))
    encoder.freeze()
elif args.model == 'Quartz':
    encoder = nemo_asr.JasperEncoder(**jasper_params["JasperEncoder"])
    fc = LinearLayer(64 * 256)  # TODO find shape from jasper_params
    fc.restore_from('./runs/{0}/checkpoints/LinearLayer-STEP-{1}.pt'.format(args.name, str(args.enc_step)))
    encoder.restore_from('./runs/{0}/checkpoints/JasperEncoder-STEP-{1}.pt'.format(args.name, str(args.enc_step)))
    encoder.freeze()

l2_regularizer = L2Regularizer()

N = len(train_data_layer)
steps_per_epoch = math.ceil(N / float(batch_size) + 1)

"""BUILDING TRAIN GRAPH"""

audio_signal, audio_signal_len, commands, command_len = train_data_layer()
processed_signal, processed_signal_len = data_preprocessor(input_signal=audio_signal, length=audio_signal_len)

encoded, encoded_len = encoder(audio_signal=processed_signal)
if args.model == 'Quartz':
    encoded = fc(encoder_output=encoded)
encoded = l2_regularizer(embeds=encoded)

"""BUILDING TEST GRAPH"""

test_audio_signal, test_audio_signal_len, test_commands, test_command_len = eval_data_layer()
test_processed_signal, test_processed_signal_len = data_preprocessor(
    input_signal=test_audio_signal, length=test_audio_signal_len
)

test_encoded, test_encoded_len = encoder(audio_signal=test_processed_signal)
if args.model == 'Quartz':
    test_encoded = fc(encoder_output=test_encoded)
test_encoded = l2_regularizer(embeds=test_encoded)

model_path = neural_factory.checkpoint_dir

print('Evaluating train set...')
train_tensors = neural_factory.infer(
    tensors=[commands, encoded],
    checkpoint_dir=model_path
)

train_y = np.concatenate(train_tensors[0])
train_embeds = np.concatenate(train_tensors[1])

if args.save:
    np.save(f'files/{args.name}_train_y', train_y)
    np.save(f'files/{args.name}_train_embeds', train_embeds)

print('Evaluating test set...')
test_tensors = neural_factory.infer(
    tensors=[test_commands, test_encoded],
    checkpoint_dir=model_path
)

test_y = np.concatenate(test_tensors[0])
test_embeds = np.concatenate(test_tensors[1])

if args.save:
    np.save(f'files/{args.name}_test_y', test_y)
    np.save(f'files/{args.name}_test_embeds', test_embeds)

train_embeds = train_embeds[:, 0, :]
test_embeds = test_embeds[:, 0, :]

knn = FaissKNeighbors(args.k)
knn.fit(train_embeds, train_y)

print('Predicting...')
preds = knn.predict(test_embeds)

print(f'Accuracy: {np.mean(preds == test_y) * 100}')
print('f1_macro', f1_score(test_y, preds, average='macro'))
if args.save:
    np.save(f'files/preds{args.name}_{args.k}.npy', preds)

"""
labels = np.array(
    ['visual', 'wow', 'learn', 'backward', 'dog', 'two', 'left', 'happy', 'nine', 'go', 'up', 'bed', 'stop', 'one',
     'zero', 'tree', 'seven', 'on', 'four', 'bird', 'right', 'eight', 'no', 'six', 'forward', 'house', 'marvin',
     'sheila', 'five', 'off', 'three', 'down', 'cat', 'follow', 'yes', 'silence'])

true_words = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence']
true_labels = [np.where(labels == x)[0][0] for x in true_words]

preds[np.where(np.isin(preds, true_labels, invert=True))] = 38
test_y[np.where(np.isin(test_y, true_labels, invert=True))] = 38

print(np.mean(test_y == preds))
"""
