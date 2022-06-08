
### Getting started


#### 1. Installing packages
Install packages required by the model.
The following packages are tested on python 3.9:
* cv2
* datasets
* transformers
* imutils
* matplotlib
* numpy
* pillow
* torch*
* torchvision

\* refer to [torch website](https://pytorch.org/) in order to install the correct GPU-enabled torch version for your system <br>

Packages required by the server:
* `pip install fastapi`
* `pip install "uvicorn[standard]"`

#### 2. Downloading the model
Download the model from [Drive](https://drive.google.com/file/d/14ISB2cFMf_PbLKmWrXTdBBJs0BMYQYqn/view?usp=sharing) and extract its contents in `./functions/model/pretrained_1/` folder

#### 3. Running the server
Then `uvicorn server:app --reload` to run the server then head to [localhost:8000](localhost:8000)

## Model
The model used for emotion classification is a ViT (Vision Transformer) trained on FER-2013 dataset. The accuracy of the model is 72% and it's still a work in progress.

## Todo
sections for
* application

backend
* remove extra data conversions
* add logging (cuda availability, model version, proc. time, ...)
* separate html image repr. conversion from pred. code
* make models/ a top level folder