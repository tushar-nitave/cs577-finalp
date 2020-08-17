## CS577-finalp
### CRNN for scene text detection

In this project we implemented a CRNN model which recognized word from a given image. We trained our model on three different 
datasets namely [IC13](https://rrc.cvc.uab.es/?ch=2&com=downloads), [IC15](https://rrc.cvc.uab.es/?ch=4&com=downloads) and [MJSynth](http://www.robots.ox.ac.uk/~vgg/data/text/).
Our model performs best on MJSynth dataset which original contains 9 million cropped images out which we used 79K images for training,
12K for testing and 15K for validation. All the code is implemented all the code in python3 and keras. 

#### Extract Data from MJSynth

MJSynth is a synthetic dataset of cropped images. It is huge dataset and due to limited computing capcity we were unable
to use complete dataset. So in order to extract desired amount of data we wrote `get_data.py` module which accepts a integer
number (number of samples to be extracted) and produces synth_data dir which contains images and synth_data.csv which contains
labels.

You can manually set the number of samples in `get_data.py` or import module and call the method `get_data()`.

execute `$ python3 get_data.py `

#### Training the model

This model was originally trained on NVIDIA(R) Tesla(TM) P100 with 16GiB of memory. For 20 epochs training time was 100 minutes
with approximately 300 seconds for each epoch.

You set number of epochs and batch size in `utils.py`.

execute `$ python3 main.py` 

#### Prediction

We have included a `prediction.py` module which can be used to evaluate the model. You directly execute this method or import
and call the method. We have to manually specifiy the dataset on which we have to perform prediction. 

For directly execution do `$ python3 prediction.py`

#### Results

##### Train Set (79424 samples)
| Dataset       | Accuracy (%)     |
| ------------- |:----------------:|
| IC13          | 1.9%             |
| IC15          | 10%              |
| MJSynth       | 53.8%            |


##### Test Set (12765 samples)
| Dataset       | Accuracy (%)     |
| ------------- |:----------------:|
| IC13          | 0.07%             |
| IC15          | 5.1%             |
| MJSynth       | 45.01%           |

#### Directory Structure
1. ./src -- Contains all the python code.
2. ./data -- You have to download data and unzip in this directory so that it can be used by training and prediction
3. ./doc -- Final report.
4. ./sources -- Reserach papers we used for implementing the project and the utility code for CNN model we used for comparing with our model.
5. ./presentation -- Slides for presenting the final project
