import random
import json
import pickle
import numpy as np
from nltk_utils import bag_of_words, tokenize, stem

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

with open('intents.json', 'r') as f:
    intents = json.load(f)
print(intents)

all_words = []
tags = []
xy = []
ignore_words = ['?', '.', '!']

for intent in intents['intents']:
    tag = intent['tag']
    # add to tag list
    if tag not in tags:
        tags.append(tag)
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = tokenize(pattern)
        # add to our words list
        all_words.extend(w)
        # add to xy pair
        xy.append((w, tag))

# stem and lower each word
# remove ignore_words and sort
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(list(set(all_words)))
tags = sorted(list(set(tags)))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

pickle.dump(all_words,open('all_words.pkl','wb'))
pickle.dump(tags,open('tags.pkl','wb'))

# create training data
training = []
output = []

# create an empty array for our output
output_empty = [0] * len(tags)

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words) 

    output_row = list(output_empty)
    output_row [tags.index(tag)] = 1

    training.append([bag, output_row])

# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)

# create train and test lists. X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])

print("Training data created")

# Create model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting and saving the model 
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('model.h5', hist)

print("model created")













