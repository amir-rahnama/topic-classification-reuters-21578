# Topic Classification for Reuters-21578 dataset

The idea is to run a classification task based on the reuters-21578 dataset. There are already many implementations of the same task
such as the Keras' implementation or Giuseppe Bonaccorso's implementation. I am for sure inspired by their idea, however my models' accuracy
is reaching more than 99% accuracy on the same task. I do believe that there should be a mysterious bug in Giuseppe Bonaccorso's implementation.



### Get started

In order to run the project, fist install all the pip requirements:

```bash
pip install -r requirements.txt
```


After that, run the training functions:

```
python train.py

```

You need to know that the datafiles are already checked in the project, so you don't need to worry about configurations.


### Highlights

#### Reading data from scratch
I decided to read the data myself with python code. I know that Keras comes with the data, but it is usuall intersting to do it yourself.
Of course this comes up with a trade-off that you cannot really spend that time on analyzing trials for the model, but I made that choice and I am
kinda happy about it, like this -> :happy

#### Feature Selections
The features are Word2Vecs, I mean you should agree that there is no way you can avoid using word embeddings these days and they are such a powerful
tool. So I went for it. Before the word2vec, I am using basic tokenization and lemmatization, so nothing fancy their neither.


#### Training
First off, the results are amazing, if they are true. The reason I am adding 

>if they are true

is because over-fitting is a major problem. I need to go through the training, but my main hunch is that the reasons are:

* I am not throwing any data before feeding it to Word2Vec. So basically there is no threshold during tokenization and I am using all that data

* I am using a lot of features. You should accept that putting a threshold of 500 for each piece creates a lot of information 
and context for the model to learn.

* I am optimizing using Adam, the slow guy who takes a lot of time to update, but at the end he will not disaapoint, if you are patient with him.



The model is using a 2-layer LSTM with Adam as it's optimizer. Loss function is the usual binary cross entropy, nothing fancy. I used 
Adam because I know that while it's a slow optimizer, it converges better when it is seeing non-convex regions in the loss surface. 
So it's a safe choice, I usually prefer Adagrad because it can get stuck and makes the training faster, but it runs faster when you 
have a lot of data.




### Tensorboard

At the end, I would like to add some basic tensorboard visualizations that showed nothing interseting that I did not knew before, but one thing.
The insane progress after the first epoch and the beginning of the second epoch. I am pretty sure that the model is over-fitting.
To me, that is as bad as having a low accuracy model. It should feel the same for true Machine Learning folks, however in the age of deep learning,
over-fitting is kinda more accepted (ehem!)






### Future Work

Oh, many many interesting things to be done, still. I will just put them here in  a list:

* Flexible reading of files using cmd arguments
* Caching data (I have coded part of caching the data, but it has a :bug:) 
* Hyperparams of the model should be using hyperopt and not configured only by hand
* I think that LSTM is over-fitting. Maybe reducing the number of neurons in the layer or using a Feedforward Neural Network might help. 
* Ipython Notebook visualizations that can show embeddings, and some basic exploratory analysis work


I decided to keep the repo alive after my experiment and more to come soon, after my cold got better!







