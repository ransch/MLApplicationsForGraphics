# ML Applications For Graphics - Final Project

This repository contains the final project of the workshop
"ML Applications For Graphics".

## Project Structure

* */src* contains training and evaluation source files
    * */src/clustering* contains source files that should be used for splitting the training set into subsets
    * */src/networks* contains the source code for the neural networks used in our experiments
    * */src/training* contains all the source files used for training the model
    * */src/evaluation* contains source files used for experiments
    * */src/hyperparameters.py* is a file which contains all the hyperparameters used in the project
    * */src/settings.py* is a file which contains environment-related asserts and definitions
* */scripts* contains bash scripts for settings up the environment
* */clustering* and */models* contains different trained models, their hyperparameters and the results of the experiments conducted

## Training a Model

For training a model, several steps should be followed:

1. The training data should be downloaded; In our experiments, we've used Jon Shamir’s frogs dataset which consists of almost eight thousand 64*64 images. It is available on [Github](https://github.com/jonshamir/frog-dataset).

2. The training data should be split into subsets by running the files under */src/clustering*; The subsets are serialized to the path defined in */src/settings.py*.

3. The algorithm should be run by running the files under */src/training*: */src/training/trainGLO* trains the first and the third GLO variants, */src/training/trainGLO2Subsets* trains the second GLO variant, and */src/training/trainEnc* trains an Encoder. You should configure the */src/hyperparameters.py* and */src/settings.py* before.
