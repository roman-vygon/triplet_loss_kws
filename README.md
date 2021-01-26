# Learning Efficient Representations for Keyword Spotting with Triplet Loss

Code for the paper [Learning Efficient Representations for Keyword Spotting with Triplet Loss](https://arxiv.org/abs/2101.04792) \
by Roman Vygon and Nikolay Mikhaylovskiy

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

* [NeMo v0.10.1](https://github.com/NVIDIA/NeMo/tree/v0.10.1)
* [PyTorch 1.5.0](https://pytorch.org/get-started/previous-versions/)
* [FAISS](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md)
* [TextGrid](https://pypi.org/project/TextGrid/)

```
Give examples
```

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Training
To train the triplet encoder run:
```
python TripletEncoder.py --name=test_encoder
```
To train a no-triplet model, or to train a classifier based on the triplet encoder run:
```
python TripletClassifier.py --name=test_classifier
```
You can use ```--help``` to view the description of arguments

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc

## Additional files
Data manifests, librispeech alignments and distance measures can be found [here](https://drive.google.com/drive/folders/13pDTAPn0fzJ2Q4IOmHobry3UsDtVz4ro?usp=sharing)
