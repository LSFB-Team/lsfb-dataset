# LSFB Dataset

This library is a companion for the [French Belgian Sign Language dataset](https://lsfb.info.unamur.be/). You will find useful functions to load and manipulate the video from the LSFB dataset. The package provide a pytorch dataset class and several useful transformations methods for video.

A complete documentation for this library is available [here](https://jefidev.github.io/lsfb-dataset/)

# Project Testing and Deploy

This library is officialy deploied on [PyPI](https://pypi.org/project/lsfb-dataset/). This means that, for once (don't lie), the code must be tested and validated before deployment.

This section will explain how to run the test suite, how to setup a local environment enabling you to use the library as if it was installed from PyPi and how to finally deploy when all is good.

In a near future, all the process should be automatised when a pull request is accepted on master.

## Write and Run Tests

This project use the [tox](https://tox.readthedocs.io/en/latest/) and [pytest](https://docs.pytest.org/en/latest/) for testing. Tox is a tool for running tests in multiple environments while pytest is the most commonly used library for writting test suites. The configuration of the tox environments are located in the `tox.ini` file.

To run the test suite, you just have to run the command `tox` in the lsfb-dataset directory. If some dependencies were added to the `setup.py` file, you need to run `tox --recreate` in order to force recreating the test environment including that dependency.

Writting good test is trickier, please refer to the [pytest documentation](https://docs.pytest.org) for more information.

## Local Install

You can always install the library locally by running the command `python -m pip install ./lsfb-dataset` in the root directory.

# Sources 

[Testing your python package as installed](https://blog.ganssle.io/articles/2019/08/test-as-installed.html)
[Testing & Packaging](https://hynek.me/articles/testing-packaging/)