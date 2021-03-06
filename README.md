# Deep Reinforcement Learning Nanodegree

This repository contains my **Deep Reinforcement Learning** solution to one of 
the stated problems within the scope of the following Nanodegree:

[Deep Reinforcement Learning Nanodegree Program - Become a Deep Reinforcement Learning Expert](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893)



Deep Reinforcement Learning is a thriving field in AI with lots of practical 
applications. It consists of the use of Deep Learning techniques to be able to
solve a given task by taking actions in an environment, in order to achieve a 
certain goal. This is not only useful to train an AI agent so as to play videogames, 
but also to set up and solve any environment related to any domain. In particular, 
as a Civil Engineer, my main area of interest is the **AEC field** (Architecture, 
Engineering & Construction).

## Navigation Project

This project consists of a Navigation problem, which is contained within the Value 
Based Methods chapter and is solved by means of Deep Q Learning algorithms.

### Environment description

The environment used for this project is based on the *Unity ML-Agents* Banana Collector 
environment from 2018. Nonetheless, the updated corresponding equivalent in 2022 
would be the
[Food Collector Environment](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Learning-Environment-Examples.md#food-collector).

It consists of a square playground in which there exist 
both yellow and blue bananas. An agent must then pick the yellow ones, while discarding
the blues.

![alt text](https://video.udacity-data.com/topher/2018/June/5b1ab4b0_banana/banana.gif)
<br/>

Its characteristics are as follows:
- The state space has 37 dimensions and contains the agent's velocity, along with 
ray-based perception of objects in front of it.
- The action space consists of 4 discrete actions. Namely: move forward, move 
backward, turn left and turn right.
- A reward of +1 or -1 is provided when a yellow or blue banana is collected, 
respectively. 

### Getting Started

Firstly, the reader is advised to set up the 
[Food Collector Environment](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Learning-Environment-Examples.md#food-collector)
instead of the legacy Banana Collector, specially if not following the Nanodegree. This will allow to install recent 
versions of the libraries and potentially avoid some unfixed bugs from old versions.

#### Installing required dependencies

Having said that, the exact environment used in this repository for the scope of the Nanodegree can be cloned
from the following repository, in which there are also the instructions to install the required dependencies:
[udacity/Value-based-methods](https://github.com/udacity/Value-based-methods)

The previous cloning has been *Step 1*. In addition, the compressed file corresponding to the user operating system 
must be downloaded from one of the links below, and placed inside the `./p1_navigation/` folder after unzipping it, which 
constitutes *Step 2*:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

Once the two previous steps are accomplished, the following step (*Step 3*) is to clone the current repository and place 
the folder at the root of the freshly configured *Value-based-methods*. From there, it only remains to change the 
environment path to match the file downloaded in *Step 2*. This path can be changed at 
[the following line of environment.py](https://github.com/cvillagrasa/DeepReinforcementLearning_Navigation/blob/c3b8224bd47d720f740136269a0f198f5f20c1f9/environment.py#L12).

At this point, *Step 4* consists of installing Seaborn within the Python environment by running `pip install seaborn`.

And that's it. After those four steps, the Jupyter Notebook *Navigation.ipynb* from this repository can already 
be executed.

### Solution

My solution to the posed problem can be found in the 
[Solution Report](https://htmlpreview.github.io/?https://github.com/cvillagrasa/DeepReinforcementLearning_Navigation/blob/master/Report.html), 
as well as in the code included 
within this repository.

It is worth stating that it has been a fascinating exercise which has let me further understand the dynamics of the 
Deep Q Learning algorithm, along with three of its variants.
