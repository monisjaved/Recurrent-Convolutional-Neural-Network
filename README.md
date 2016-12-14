# rcnn-text-classificaiton
The model developed in this project uses combination of Recurrent and Convolution Neural Network for tweet classification. The model automatically captures important features and trains model accordingly.


## Dependencies 
- numpy
- theano
- twokenize
- word2vec

## Dataset
Database of 10000 tweets is used for creating word embeddings using word2vec. A subset of that data of size 1532 is used for training and testing. Training is done with 1000 tweets and testing is done with 532 tweets set by randomly dividing the dataset.

## Training
Training is carried using GPU, Nvdia GeForce 750M

Before training create initial word embeddings from your text corpse using word2vec.

```python
import word2vec
#
# tweet-corpse.txt contains 10K tweets in lower case.
# Size 100 defines the vector size of the word embedding
word2vec.word2vec('data/tweet-corpse.txt', 'data/training_vec.bin', size=100, verbose=True)
```

We can then start training by

```
sudo python rcnn_theano
```

Manually, Once we have vector embedding, we can continue to load word2vec data with TweetPreProcess and continues training

```python
t = TweetPreProcess() # Loads dictionary and its word embedding
x, y = t.create_set(filename="data/1700.csv", random=True) # Load dataset into x and y
```

Once we load data we can divide the data into various set and then start training using `RCNN` class.

```python
# Sharing word embeddings to theano shared memory to sothat it also can be updated
vector_dict = theano.shared(np.asarray(t.word_vec).astype("float32"), name="vector_dict")
# Dividing into training and validation set
train_x_set = [x[i][:] for i in range(1000)]
train_x = [y[i][:] for i in range(1000)]
valid_x_set = [x[i][:] for i in range(1000, len(x))]
valid_x = [y[i][:] for i in range(1000, len(x))]
# Initializing using vector_dict
model = RCNN(vector_dict)
#
# Training 
model.process_set(train_x_set,
                  train_x,
                  num_epochs=64,
                  verbose=True,
                  verbose_interval=1,
                  file_name="model_ada.model",
                  valid_x=valid_x_set,
                  valid_y=valid_x)
```

## Prediction
Once the model is trained you can use to predict new data 
```python
model.prediction(t.sentence2index("Flipkart has great service "), 4) # Second argument is length of string
```

or you can load best validation saved model and then predict
```python
model = RCNN(file_name= "model_ada")
model.prediction(t.sentence2index("Flipkart has great service"), 4)
```

## Results
Training this model on a 1000 tweets data and testing the best stable model on 532 tweets gives accuracy of 80%

Log files of the project is at model/report.txt along with best validation model.

## Author
Moonis Javed


## Version
1.0.0
