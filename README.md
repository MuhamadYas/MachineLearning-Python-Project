# Scratch Detection on Wafer Maps
## Overview
This repository contains a solution to the Scratch Detection Assignment from NI’s Data Science team. The task is to build an automated model that can flag dies on a semiconductor wafer that belong to a physical scratch. Scratches appear as elongated clusters of faulty dies and may include a few “good” dies that should still be discarded (inked). Identifying these patterns manually is time‑consuming and error‑prone, so the objective is to train a machine‑learning model to predict scratch dies for new wafer maps.

The dataset is provided as a logical wafer map rather than an image. Each row describes a single die using the following fields:

  WaferName – unique wafer identifier.
  DieX – horizontal coordinate of the die on the wafer.
  DieY – vertical coordinate of the die on the wafer.
  IsGoodDie – whether the die passed electrical tests (Boolean).
  IsScratchDie – whether the die belongs to a scratch (Boolean).

Our goal is to learn from labelled training wafers and predict the IsScratchDie label for each die in an unseen test set. The business drivers for this project include automation of a costly manual process, improving overall quality and yield, and enabling wafer‑level scratch classification. Low‑yield wafers (with many faulty dies) are excluded from scratch detection because random clusters may otherwise be mistaken for scratches.

## Solution Approach
The provided notebook implements an end‑to‑end pipeline using a convolutional neural network (CNN) to segment scratch regions on a wafer map. Below is a high‑level overview of the steps taken:

1. Data Pre‑processing:

Filtering low‑yield wafers: Wafers with yield below a user‑defined threshold (default 0.9) are removed. Yield is computed as the proportion of good dies (IsGoodDie = True) per wafer. This follows the business rule to skip scratch detection on low‑yield wafers.

Wafer normalisation: Each wafer is converted into a fixed‑size image (71×71). The choice of 71×71 accommodates the largest wafer in the training set; smaller wafers are padded. Three channels are used:

  A binary mask of good/bad dies (IsGoodDie).

  Normalised x coordinates.

  Normalised y coordinates.

Target masks: A single‑channel mask stores the scratch labels (IsScratchDie) for training.

2. Model Architecture

A CNN with dilated convolutions is built using TensorFlow/Keras. The network stacks regular 3×3 convolutions with dilation rates of 2 and 4 to capture elongated scratch patterns. A final 1×1 convolution with sigmoid activation outputs per‑pixel probabilities.

A custom masked loss function (binary cross‑entropy) ignores padded regions where there is no die.

3. Training

The training script samples a fraction (e.g., 20 %) of wafers to create a mini‑dataset and trains the CNN for 30 epochs with a batch size of 4. The model is saved as scratch_detector.keras for later inference.

4. Prediction & Post‑processing

For each wafer in the test set, the model predicts a scratch mask. Predictions greater than 0.5 are considered scratches.

Inked dies expansion: After predicting scratches on bad dies, morphological dilation (using OpenCV) expands the scratch region to include neighbouring good dies (inked dies). This step reflects the business rule that good dies on a scratch should also be discarded.

The final per‑die predictions are stored in cnn_predictions_padded.csv. A helper function is provided to visualise wafer maps and compare inputs and predictions.

5. Pipeline Execution

A run_padded_pipeline() function filters training and test wafers, trains the model, predicts scratch dies on the test set, visualises some results and reports completion. A final step writes your name and email along with the prediction file name.

## Requirements
The solution uses Python 3 with the following key libraries:

pandas and numpy for data handling.

matplotlib for plotting wafer maps.

opencv-python (cv2) for morphological operations.

tensorflow/keras for the CNN implementation.
