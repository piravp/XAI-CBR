import pickle

from sklearn.feature_extraction.text import HashingVectorizer

# Naive bayes and decision trees.
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

# Stopwords
import nltk
from nltk.corpus import stopwords

def part1():
    # Download stopwords if not availible
    print("PART 1")
    try:
        stopwords.words("english")
    except Exception as e: # download stopwords for nltk if we can't find the file.
        print(e)
        nltk.download('stopwords')
    finally:
        # collect stopwords
        stop_words = set(stopwords.words("english"))
    # Load data
    data = pickle.load(open("sklearn-data.pickle","rb"))

    vectorizer = HashingVectorizer(stop_words=stop_words,n_features=2**14, lowercase = True, binary=True)

    # Vectorize both test and train datasets.
    X = vectorizer.transform(data["x_train"])
    X_test = vectorizer.transform(data["x_test"])

    decisionTree(X,data["y_train"],X_test,data["y_test"])
    naiveBayes(X,data["y_train"],X_test,data["y_test"])
    


def naiveBayes(X, y, X_test, y_test):
    classifier = BernoulliNB()
    print(classifier)
    classifier.fit(X=X,y=y)
    y_pred = classifier.predict(X_test)
    score = accuracy_score(y_true=y_test,y_pred=y_pred)*100
    print("NaiveBayes: {0:.2f} %".format(score))

def decisionTree(X, y, X_test, y_test):
    classifier = DecisionTreeClassifier(max_depth=10)
    print(classifier)
    classifier.fit(X=X,y=y)
    y_pred = classifier.predict(X_test)
    score = accuracy_score(y_true=y_test,y_pred=y_pred)*100
    print("decisionTree: {0:.2f} %".format(score))

from tensorflow import keras
from keras import preprocessing,layers

def part2(fit_generate=True):
    print("PART 2")
    data = pickle.load(open("keras-data.pickle","rb"))

    # change to speed up training
    #max_length = 384#data["max_length"]
    max_length = min(800, data["max_length"]) # 
    print("using max_length",max_length,"of", data["max_length"])

    vocab_size = data["vocab_size"]

    #sequence.pad sequences
    x_train = preprocessing.sequence.pad_sequences(data["x_train"], maxlen=max_length)
    x_test = preprocessing.sequence.pad_sequences(data["x_test"], maxlen=max_length)

    y_train = np.array(data["y_train"])
    y_test = np.array(data["y_test"])


    # Model init
    model = keras.Sequential()

    """ Model 1
    #model.add(keras.layers.Embedding(input_dim=vocab_size+1, output_dim=128, input_length=max_length, mask_zero=True)) # Embed the input vector to smaller dimention.
    #model.add(keras.layers.LSTM(96, dropout=0.2) 
    #model.add(keras.layers.Dense(1, activation='sigmoid')) # Final layer.
    """ 

    
    """ Model 2
    model.add(keras.layers.Embedding(input_dim=vocab_size+1, output_dim=256, input_length=max_length, mask_zero=True)) # Embed the input vector to smaller dimention.
    model.add(keras.layers.LSTM(256, dropout=0.2, bias_regularizer=keras.regularizers.l2()))
    model.add(keras.layers.Dense(1,activation='sigmoid')) # Final layer.

    """

    # Final model, with the best results.
    model.add(keras.layers.Embedding(input_dim=vocab_size+1, output_dim=256, input_length=max_length, mask_zero=True)) # Embed the input vector to smaller dimention.
    model.add(keras.layers.LSTM(256, dropout=0.2, bias_regularizer=keras.regularizers.l2(), 
    activity_regularizer=keras.regularizers.l2(), kernel_regularizer=keras.regularizers.l2(),
    recurrent_regularizer=keras.regularizers.l2())) # learn the sequence importance (hopefully) # 160
    model.add(keras.layers.Dense(1,activation='sigmoid')) # Final layer.

    print(model.summary())


    optim = keras.optimizers.Adam(lr=0.001)
    model.compile(loss="binary_crossentropy", optimizer=optim, metrics=["accuracy"])

    # Basicly set batch_size to remaining memory to maximize training efficiency. 

    if(fit_generate): #didn't help
        model.fit_generator(generator(x_train, y_train, batch_size=200), validation_data=generator(x_test, y_test, batch_size=200), validation_steps=20,
            steps_per_epoch=30, epochs=30)
    else:
        # train, with validation split, ideely use from test data, but require too much memory
        #model.fit(generator(x_train, y_train,batch_size=200), validation_data=generator(x_test, y_test, batch_size=200), validation_steps=20, 
        #epochs=20, steps_per_epoch=20, callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss",mode='max',patience=3,verbose=1)]) 

        model.fit(generator(x_train, y_train,batch_size=200), validation_data=generator(x_test, y_test, batch_size=200), validation_steps=30, 
        epochs=100, steps_per_epoch=50, callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss",mode='min',patience=3,verbose=1),]) 

        # If we would want to store the best version.
        #keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    
    # Evaluate performance of network
    score, acc = model.evaluate(x_test, y_test, batch_size=200)
    
    print("LSTM: {0:.2f} %".format(acc*100))

import numpy as np
# Data generator for fit and evaluate
def generator(x, y, batch_size):
    while True: # never ending loop
        #Randomly select batch_size number of sentences
        indx = np.random.choice(len(x), batch_size, replace =False)
        yield x[indx], y[indx] # return selected data and corresponding labes


# Run parts
part1()
#part2(fit_generate=False)