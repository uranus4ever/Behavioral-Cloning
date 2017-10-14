# **Behavioral Cloning** 


---

### **Overview**

This project aims to clone human driving behavior in the [simulator](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4419_windows-sim/windows-sim.zip) with a Deep Learning Neural Network, namely end to end learning. During the training phase, I used keyboard to drive the car on Track1. Then the images and steering angles recorded are the input to feed the neural network model. After then in autonomous mode, the model predicted steering angle accoring to images the car "saw" and self steered. Finally the model performance is tested on both tracks as the following animations, it can self-drive stably and endlessly on tracks.

| Track 1 - Training | Track 2 - Validation|
| :-: | :-: |
| ![alt text][gif1] | ![alt text][gif2] |
| ![Track1.mp4][video1] | ![Track2.mp4][video2] |


[//]: # (Image References)

[image1]: ./img/track_view.png "Two Tracks"

[image2]: ./img/three_camera_view.png "Three Camera View"

[image3]: ./img/img_process.png "Image Process"

[image4]: ./img/NVIDIA_model.png "Model Architecture"

[image5]: ./img/model_evaluation_balance.png "Model Accurency"

[gif1]: ./img/Track1_gif.gif "Track1 gif"

[gif2]: ./img/Track2_gif.gif "Track2 gif"

[image8]: ./img/Steering_angle_distribution.png "Steering Angle Distribution"

[video1]: ./Track1.mp4 "Track1 video"

[video2]: ./Track2.mp4 "Track2 video"

---

### Usage

#### 1. Content

My project includes the following files:

* `model.py` containing the script to create and train the model

* `drive.py` for driving the car in autonomous mode

* `model_balanced.h5` and `model_balanced.json` for a trained convolution neural network with weight

* ` Track1.mp4` and `Track2.mp4` showing self-driving mode


#### 2. How to run the code

Using the Udacity provided simulator and my ```drive.py``` file, the car can be driven autonomously around the track by executing 

```sh
python drive.py model_balanced.json run1
```
The simulator provides two tracks, lake track (easy mode) and hill track (hard mode).

![alt text][image1]

#### 3. Dependencies
* Keras
* TensorFlow
* OpenCV
* NumPy
* SciPy
* sklearn

### Model Architecture and Training Strategy

#### 1. Appropriate Training Data

Training data was collected in the simulator run by myself. I collected over **one hour equivalent time** of driving on Track 1, which contained about 25k data and 75k images. In terms of driving mode, it contained both **smooth** driving style (mostly stay in the middle of the road) and **recovery** driving style (drive off the middle and then steer to the middle).

I used a combination of center camera, left and right camera randomly with `select_img` function, and use `angle_correction=0.23` to correct left or right camera.

After multiple model training, it is found that in autonomous mode, the car performas far from expected to pull back to the center if off the middle. Its root cause is the **uneven** training data. Concretely, most of (>90%) steering angles are close to zero, so trained model could not predict a correct turn angle when needed. To combat with that, I extract the angle data (range[-1, 1]), absolute and balance them with the following code (inspired by [navoshta](https://github.com/navoshta/behavioral-cloning)):

```
num_bins = 500 # interval in (0,1)
bin_n = 300  # Maximum number in each bin
balance_box = []
start = 0
for end in np.linspace(0, 1, num=num_bins)[1:]:
    idx = (angles >= start) & (angles < end)
    n_num = min(bin_n, angles[idx].shape[0])
    sample_idx = sample(range(angles[idx].shape[0]), n_num)
    lines_range = np.array(lines)[idx].tolist()
    for i in range(len(sample_idx)):
        balance_box.append(lines_range[sample_idx[i]])
    start = end
```
After this balance data filter, there are 6824 data left, distributing as follows:

![alt text][image8]

#### 2. Data Augment and Pre-Process

In order to increase model prediction accurancy and reduce training time, I pre-processed the images before feeding into the model with multiple techniques. 

* **Crop**. I cropped the pixels in the top and bottom of the image, which contain mostly trees and hills, helpless to train the model.

* **Random Gamma**. To make algorithm robust to deal with frames with dark background color, brightness is randomly reset with this [reference](http://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/).
```
def random_gamma(image):
    gamma = np.random.uniform(0.4, 1.5)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)
```
* **Random Shadow**. Shadow is the main difficulty to mislead the car to steer. To combat with that, apply random shadow in the training set to mimic various kinds of shadows.

* **Flip**. Track 1 is counter-clock based, namely left turn domined. Then I flipped images randomly with Bernoulli distribution. 
```
def random_flip(image, steering_angle, flipping_prob=0.5):
    head = bernoulli.rvs(flipping_prob)
    if head:
        return np.fliplr(image), -1 * steering_angle
    else:
        return image, steering_angle
```

* **Resize**. To speed up training, I resized all images with 32x128.

![alt text][image3]

#### 3. Model Architecture

Inspired by NVIDIA's [paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) "End to End Learning for Self-Driving Cars", my model architecture was similar to the following:

![alt text][image4]

The structure is not complex and works pretty well. To avoid overfitting, I applied dropout technique.
In addition, **generator** is used to process a large amount of data batch by batch. And I ran the model training on [Floydhub](https://www.floydhub.com) for GPU cloud computing.
In terms of model save function, thanks to [upul's code](https://github.com/upul/Behavioral-Cloning/blob/master/helper.py).

#### 4. Model Evaluation

In order to gauge how well the model was working, I split 20% training data into validation set. The mean square error loss figure shows the training loss is very close to validation loss after 10 epochs, which means the model is neither underfit nor overfit. 

![alt text][image5]

### Reflection

In real human driving, there are only two inputs need to be controlled - speed (throttle and brake) and steering angle. Behavioral cloning is an amazing idea to teach machine how to self-drive with neural networks to control these two parameters. Back to this project, the followings are worthy to be improved:

 - The performance on Track2 is not as good as on Track1, due to much more hard turns, complex shadows and dark light. To improve that, more targeted data augment techniques and more training data with hard turn need to be taken.
 - Data augment needs to be further explored. This technique can increditibly save training time, if in reality, saving development time and cost.

When it comes to extensions and future directions, I would like to highlight the followings:

 - In current simulator autonomous mode, only steering angle is learned. To step further, throttle control could also be learned from human behaviour. (At present it is controled by PID)
 - Train a model in real road driving with a new simulator.
 - Experiment with other Neural Network models, for example, Recurrent Neural Network or Reinforcement Learning structure.

