# Learning Efficient Representations for Keyword Spotting with Triplet Loss

Code for the paper [Learning Efficient Representations for Keyword Spotting with Triplet Loss](https://arxiv.org/abs/2101.04792) \
by Roman Vygon(roman.vygon@gmail.com) and Nikolay Mikhaylovskiy(nickm@ntr.ai).

### Prerequisites

* [NeMo v0.10.1](https://github.com/NVIDIA/NeMo/tree/v0.10.1)
* [PyTorch 1.5.0](https://pytorch.org/get-started/previous-versions/)
* [FAISS](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md)
* [TextGrid](https://pypi.org/project/TextGrid/)

## Training
To train a triplet encoder run:
```
python TripletEncoder.py --name=test_encoder --manifest=MANIFEST --model=MODEL 
```
To train a no-triplet model, or to train a classifier based on the triplet encoder run:
```
python TripletClassifier.py --name=test_classifier --manifest=MANIFEST --model=MODEL
```
You can use ```--help``` to view the description of arguments.

#### Hardware Requirements
Training was performed on a single Tesla K80 12GB.
| Model         | Batch Size     | VRAM  |
| ------------- |:-------------:| -----:|
| Res15         | 35*4 | 11GB |
| Res8      | 35*10      |   4GB |


## Testing

To test a triplet encoder run:
```
python infer_train.py --name=test_encoder --manifest=MANIFEST --model=MODEL --enc_step=ENCODER_TRAINING_STEP
```
To test a classifier-head model run:
```
python infer_notl.py --name=test_encoder --cl_name=test_classifier --manifest=MANIFEST --model=MODEL --enc_step=ENCODER_TRAINING_STEP --cl_step=CLASSIFIER_TRAINING_STEP
```
You can use ```--help``` to view the description of arguments.
## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Datasets

### LibriSpeech
You can download the test-clean-360 here: http://www.openslr.org/12.
If the site doesn't load see [this code](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/audio/librispeech.py) for direct links to the files.

### Google Speech Commands 
Use [this](https://github.com/NVIDIA/NeMo/blob/v0.10.1/examples/asr/notebooks/3_Speech_Commands_using_NeMo.ipynb) notebook to download and prepare the Google Speech Commands dataset.
## Additional files
~~Data manifests, librispeech alignments and distance measures can be found [here](https://drive.google.com/drive/folders/16jcbL3yPvFfZphL2VFg2mDcruh5KDXP2). You'll need to update the `manifests.json` file with the dataset path. You can convert LibriWords manifests with [convert_path_prefix.ipynb](https://drive.google.com/file/d/1X3_MacQvyCXAInMq91iDs0EQVH6MwkSQ/view?usp=sharing)~~

The files sadly went missing, I'll try to recover them, if anyone had a chance to download them please contact me.
