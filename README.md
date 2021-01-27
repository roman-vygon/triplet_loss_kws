# Learning Efficient Representations for Keyword Spotting with Triplet Loss

Code for the paper [Learning Efficient Representations for Keyword Spotting with Triplet Loss](https://arxiv.org/abs/2101.04792) \
by Roman Vygon and Nikolay Mikhaylovskiy

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
You can use ```--help``` to view the description of arguments

### Testing

To test a triplet encoder run:
```
python infer_train.py --name=test_encoder --manifest=MANIFEST --model=MODEL --enc_step=ENCODER_TRAINING_STEP
```
To test a classifier-head model run:
```
python infer_notl.py --name=test_encoder --cl_name=test_classifier --manifest=MANIFEST --model=MODEL --enc_step=ENCODER_TRAINING_STEP --cl_step=CLASSIFIER_TRAINING_STEP
```
You can use ```--help``` to view the description of arguments
## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Additional files
Data manifests, librispeech alignments and distance measures can be found [here](https://drive.google.com/drive/folders/13pDTAPn0fzJ2Q4IOmHobry3UsDtVz4ro?usp=sharing)
