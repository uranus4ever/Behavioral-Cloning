#**Behavioral Cloning** 


---

**Overview**

This project aims to clone human driving behavior in the [simulator](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4419_windows-sim/windows-sim.zip) with a Deep Learning Neural Network. During the training phase, I used keyboard to drive the car. Then the images and steering angles recorded are the input to train the model. After then in autonomous mode, the model predicted steering angle accoring to images the car "saw" and self drove. Finally the model performance is tested on both tracks as the following animations.

| Track 1 | Track 2|
| :-: | :-: |
| ![alt text][image6] | ![alt text][image7] |


[//]: # (Image References)

[image1]: ./img/track_view.png "Two Tracks"

[image2]: ./img/three_camera_view.png "Three Camera View"

[image3]: ./img/img_process.png "Image Process"

[image4]: ./img/NVIDIA_model.JPG "Model Architecture"

[image5]: ./img/model_evaluation.png "Model Accurency"

[image6]: ./img/Track1_gif.gif "Track1 gif"

[image7]: ./img/Track2_gif.gif "Track2 gif"

---

###Usage

####1. Content

My project includes the following files:

* ```model.py``` containing the script to create and train the model

* ```drive.py``` for driving the car in autonomous mode

* ```model.h5``` containing a trained convolution neural network 

* ```README.md``` summarizing the results

* ``` Track1.mp4``` and ```Track2.mp4``` showing self-driving mode


####2. How to run the code

Using the Udacity provided simulator and my ```drive.py``` file, the car can be driven autonomously around the track by executing 

```sh
python drive.py model.json file_name
```
The simulator provides two tracks, lake track (easy mode) and hill track (hard mode).

![alt text][image1]

####3. Dependencies
* Keras
* TensorFlow
* OpenCV
* NumPy
* SciPy
* sklearn

###Model Architecture and Training Strategy

####1. Data Augment and Pre-Process

In order to increase model prediction accurancy and reduce training time, I pre-processed the images before feeding to the model with multiple techniques. 

* **Crop**. I cropped the pixels in the top and bottom of the image, which contain mostly trees and hills, helpless to train the model.

* **Random Gamma**. To make algorithm robust to deal with shadows, brightness is randomly reset with this [reference](http://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/)
```
def random_gamma(image):
    gamma = np.random.uniform(0.4, 1.5)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)
```
* **Sharpen**. Guassian blur is used to sharpen the images. 
```
def blur(img):
    gb = cv2.GaussianBlur(img, (5,5), 20.0)
    return cv2.addWeighted(img, 2, gb, -1, 0)
```
* **Flip**. I noticed Track 1 is counter-clock based, namely left turn domined. Then I flipped images randomly with Bernoulli distribution. 
```
def random_flip(image, steering_angle, flipping_prob=0.5):
    head = bernoulli.rvs(flipping_prob)
    if head:
        return np.fliplr(image), -1 * steering_angle
    else:
        return image, steering_angle
```

* **Resize**. To speed up training, I resized all images with 32x128.

####2. Appropriate Training Data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, left and right sides of the road randomly with ```select_img``` function, and use ```angle_correction=0.23``` to correct left or right camera.

I collected about 40 minutes of driving on Track 1. It contains both **smooth** driving style (mostly stay in the middle of the road) and **recovery** driving style (drive off the middle and then steer to the middle).


####3. Model Architecture

Inspired by NVIDIA's [paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) "End to End Learning for Self-Driving Cars", my model architecture was similar to the following:

![alt text][image4]

In addition, **generator** is used to process a large amount of data batch by batch.

####4. Model Evaluation

Over 20k training data was collected. To combat overfit, I shuffle the data before feeding into the model. In order to gauge how well the model was working, I split 20% training data into validation set. The mean square error loss figure shows the training loss is very close to validation loss after 5 epochs, which means the model is neither underfit nor overfit. 

![alt text][image5]

###Reflection

In real human driving, there are only two inputs need to be controlled - speed (throttle and brake) and steering angle. Behavioral cloning is the amazing idea to teach machine how to self-drive with Neural Networks to control these two parameters. Back to this project, the followings are worthy to be improved:

 - Most of the steering angles in the training are close to 0 because human can look ahead the track and keep car under control. But it helps less for model prediction as it can only "see" the current view.
 - Data augment needs to be further explored. This technique can increditibly save training time, in other words, if in the real life, saving development time and cost.

When it comes to extensions and future directions, I would like to highlight the followings:

 - Train a model in real road driving with a new simulator.
 - Experiment with other Neural Network models, for example, Recurrent Neural Network or Reinforcement Learning structure.

