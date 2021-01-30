from functools import partial
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from nemo import logging
from nemo.backends.pytorch import DataLayerNM
from nemo.collections.asr.parts.dataset import AudioLabelDataset, seq_collate_fn
from nemo.collections.asr.parts.features import WaveformFeaturizer
from nemo.collections.asr.parts.perturb import AudioAugmentor
from nemo.collections.asr.parts.perturb import perturbation_types
from nemo.core.neural_types import *
from torch.utils.data.sampler import BatchSampler


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples, class_dists, class_probs, probs_num):
        self.labels = torch.tensor(labels)
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes
        self.class_dists = class_dists
        self.class_probs = class_probs
        self.probs = [[1., 0., 0.],
                      [0., 1., 0.],
                      [0., 0., 1.],
                      [0.5, 0.5, 0.],
                      [0.5, 0., 0.5],
                      [0., 0.5, 0.5],
                      [0.33, 0.33, 0.33]]
        self.probs = self.probs[probs_num]

    def pick_nearby(self, label_set, n_classes, class_dists):
        with torch.no_grad():
            first_labels = np.random.choice(label_set, n_classes // 2, replace=False)
            second_labels = []
            for label in first_labels:
                for sec_label in class_dists[label][np.random.randint(3):]:
                    if sec_label in label_set:
                        second_labels.append(sec_label)
                        break
            return np.concatenate([first_labels, np.array(second_labels)])

    def pick_probs(self, label_set, n_classes, class_probs):
        return np.random.choice(label_set, n_classes, p=class_probs[label_set] / np.sum(class_probs[label_set]))

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:

            chance = np.random.rand()
            if chance < self.probs[0]:
                classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            elif chance < self.probs[0] + self.probs[1]:
                classes = self.pick_nearby(self.labels_set, self.n_classes, self.class_dists)
            else:
                classes = self.pick_probs(self.labels_set, self.n_classes, self.class_probs)

            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            indices = torch.tensor(list(map(int, indices)))
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size


# Ported from https://github.com/NVIDIA/OpenSeq2Seq/blob/master/open_seq2seq/data/speech2text/speech_commands.py
class BalancedAudioToSpeechLabelDataLayer(DataLayerNM):
    """Data Layer for general speech classification.

    Module which reads speech recognition with target label. It accepts comma-separated
    JSON manifest files describing the correspondence between wav audio files
    and their target labels. JSON files should be of the following format::

        {"audio_filepath": path_to_wav_0, "duration": time_in_sec_0, "label": \
target_label_0, "offset": offset_in_sec_0}
        ...
        {"audio_filepath": path_to_wav_n, "duration": time_in_sec_n, "label": \
target_label_n, "offset": offset_in_sec_n}

    Args:
        manifest_filepath (str): Dataset parameter.
            Path to JSON containing data.
        labels (list): Dataset parameter.
            List of target classes that can be output by the speech recognition model.
        batch_size (int): batch size
        sample_rate (int): Target sampling rate for data. Audio files will be
            resampled to sample_rate if it is not already.
            Defaults to 16000.
        int_values (bool): Bool indicating whether the audio file is saved as
            int data or float data.
            Defaults to False.
        min_duration (float): Dataset parameter.
            All training files which have a duration less than min_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to 0.1.
        max_duration (float): Dataset parameter.
            All training files which have a duration more than max_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to None.
        trim_silence (bool): Whether to use trim silence from beginning and end
            of audio signal using librosa.effects.trim().
            Defaults to False.
        load_audio (bool): Dataset parameter.
            Controls whether the dataloader loads the audio signal and
            transcript or just the transcript.
            Defaults to True.
        drop_last (bool): See PyTorch DataLoader.
            Defaults to False.
        shuffle (bool): See PyTorch DataLoader.
            Defaults to True.
        num_workers (int): See PyTorch DataLoader.
            Defaults to 0.
        augmenter (AudioAugmentor or dict): Optional AudioAugmentor or
            dictionary of str -> kwargs (dict) which is parsed and used
            to initialize an AudioAugmentor.
            Note: It is crucial that each individual augmentation has
            a keyword `prob`, that defines a float probability in the
            the range [0, 1] of this augmentation being applied.
            If this keyword is not present, then the augmentation is
            disabled and a warning is logged.
    """

    @property
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal(freq=self._sample_rate)),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
            'label': NeuralType(tuple('B'), LabelsType()),
            'label_length': NeuralType(tuple('B'), LengthsType()),
        }

    def __init__(
            self,
            *,
            manifest_filepath: str,
            labels: List[str],
            batch_size: int,
            sample_rate: int = 16000,
            int_values: bool = False,
            num_workers: int = 0,
            shuffle: bool = True,
            min_duration: Optional[float] = 0.1,
            max_duration: Optional[float] = None,
            trim_silence: bool = False,
            drop_last: bool = False,
            load_audio: bool = True,
            augmentor: Optional[Union[AudioAugmentor, Dict[str, Dict[str, Any]]]] = None,
            num_classes: int = 35,
            class_dists=None,
            class_probs=None,
            probs_num=0
    ):
        super(BalancedAudioToSpeechLabelDataLayer, self).__init__()

        self._manifest_filepath = manifest_filepath
        self._labels = labels
        self._sample_rate = sample_rate

        if augmentor is not None:
            augmentor = self._process_augmentations(augmentor)

        self._featurizer = WaveformFeaturizer(sample_rate=sample_rate, int_values=int_values, augmentor=augmentor)

        dataset_params = {
            'manifest_filepath': manifest_filepath,
            'labels': labels,
            'featurizer': self._featurizer,
            'max_duration': max_duration,
            'min_duration': min_duration,
            'trim': trim_silence,
            'load_audio': load_audio,
        }
        self._dataset = AudioLabelDataset(**dataset_params)
        labels = []
        for sample in self._dataset.collection:
            labels.append(self._dataset.label2id[sample.label])
        self._dataloader = torch.utils.data.DataLoader(
            dataset=self._dataset,
            batch_sampler=BalancedBatchSampler(labels, n_classes=num_classes, n_samples=batch_size // num_classes,
                                               class_dists=class_dists, class_probs=class_probs, probs_num=probs_num),
            # TODO replace with kwargs
            collate_fn=partial(seq_collate_fn, token_pad_value=0),
            num_workers=num_workers,
        )

    def __len__(self):
        return len(self._dataset)

    def _process_augmentations(self, augmentor) -> AudioAugmentor:
        augmentations = []
        for augment_name, augment_kwargs in augmentor.items():
            prob = augment_kwargs.get('prob', None)

            if prob is None:
                logging.error(
                    f'Augmentation "{augment_name}" will not be applied as '
                    f'keyword argument "prob" was not defined for this augmentation.'
                )

            else:
                _ = augment_kwargs.pop('prob')

                try:
                    augmentation = perturbation_types[augment_name](**augment_kwargs)
                    augmentations.append([prob, augmentation])
                except KeyError:
                    logging.error(f"Invalid perturbation name. Allowed values : {perturbation_types.keys()}")

        augmentor = AudioAugmentor(perturbations=augmentations)
        return augmentor

    @property
    def dataset(self):
        return None

    @property
    def data_iterator(self):
        return self._dataloader
