# LSFB Dataset Companion Library

LSFB Dataset is a companion library for the [French Belgian Sign Language (LSFB) dataset](https://lsfb.info.unamur.be/) released by the University of Namur.
The library provides a set of tools helping to download, manipulate and visualize the data. 

This library aims to reduce drastically the time you will spend on data loading and preprocessing allowing you to focus on your research.

## Datasets

Both datasets are based on the [LSFB Corpus](https://www.corpus-lsfb.be/).
The corpus is the result of the tremendous work achieved by the members of the [LSFB lab](https://www.unamur.be/lettres/romanes/lsfb-lab) from the university of Namur.

The corpus data were sanitized in order to make them easier to use. Additional metadata were added to enhance the datasets. Two versions of the dataset are available:

- **[lsfb_isol](lsfb_isol.md)** : Suitable for the isolated sign language recognition task.
- **[lsfb_cont](lsfb_cont.md)** : Suitable for the continuous sign language recognition task.

Both datasets are based on the [LSFB Corpus](https://www.corpus-lsfb.be/) containing 40 hours of annotated and translated video.
[Mediapipe](https://mediapipe.dev/) landmarks are also available for each dataset.

![Fond Baillet Latour](ressources/img/dataset-example.jpg)

## Modules

The library offers you a set of tools designed to help you to **download** and **load** the dataset.
The available modules are : 

- [**download**](download.md) : contains utilities to download the entirety (or some parts) of the datasets.
- [**datasets**](datasets.md) : contains pre-written python [Iterators](https://docs.python.org/3/c-api/iterator.html) for both the isol and continuous datasets. They could be used to feed the data in your *machine learning* pipeline.

## Sponsors

Without our sponsors, this library would not be possible.

![UNamur](ressources/img/logo-unamur.png)
![Fond Baillet Latour](ressources/img/baillet.png)