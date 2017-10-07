#**Behavioral Cloning** 


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior

* Build, a convolution neural network in Keras that predicts steering angles from images

* Train and validate the model with a training and validation set

* Test that the model successfully drives around track one without leaving the road

* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./img/track.png "Two Tracks"

[image2]: ./img/three_camera_view.png "Three Camera View"

[image3]: ./img/img_process.png "Image Process"

[image4]: ./img/NVIDIA_model.JPG "Model Architecture"

[image5]: ./img/model_evaluation.png "Model Accurency"

[image6]: ./img/img_process.png "Image Process"

[image7]: ./img/img_process.png "Image Process"

---

###Usage

####1. Content

My project includes the following files:

* ```BehavioralCloning.py``` containing the script to create and train the model

* ```drive.py``` for driving the car in autonomous mode

* ```model.h5``` containing a trained convolution neural network 

* ```README.md``` summarizing the results


####2. Submission includes functional code

Using the Udacity provided [simulator](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4419_windows-sim/windows-sim.zip) and my ```drive.py``` file, the car can be driven autonomously around the track by executing 

```sh
python drive.py model.json
```

####3. Dependencies
* Keras
* TensorFlow
* OpenCV
* NumPy
* SciPy
* sklearn

###Model Architecture and Training Strategy

####1. Image Pre-Process

In order to increase model prediction accurancy and reduce training time, I pre-processed the images before feeding to the model with multiple techniques. First I crop the pixels in the top and bottom of the image, which contain mostly trees and hills, helpless to train the model. Secondly, I used Guassian blur to sharpen the images. Next, I noticed Track 1 is counter-clock based, namely left turn domined. Then I flipped images randomly with Bernoulli distribution. And lastly, to speed up training, I resized all images with 64x128.

```
def random_flip(image, steering_angle, flipping_prob=0.5):
    head = bernoulli.rvs(flipping_prob)
    if head:
        return np.fliplr(image), -1 * steering_angle
    else:
        return image, steering_angle
```

![alt text][image1]

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

####2. Model Architecture

Inspired by NVIDIA's [paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) "End to End Learning for Self-Driving Cars", my model architecture was similar to the following:

![alt text][image4]

####3. Model parameter tuning

As the model used an Adam optimizer, the learning rate was not set manually.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road randomly with ```select_img``` function.

Before model training starting, I shuffled and normalized data.

####5. Model evaluation

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.

![alt text][image5]

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.



####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to steer back to the center if the vehicle drifts off to the side. These images show what a recovery looks like starting from ... :

![alt text][image3]

![alt text][image4]

![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would help with left turn bias. For example, here is an image that has then been flipped:

![alt text][image6]

![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 


I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.