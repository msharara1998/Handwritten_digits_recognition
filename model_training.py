import ANN
from keras.datasets import mnist
import pickle

#create the model
model = ANN.ANN([784,40,20,10])

#import the data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#preprocess the data
training_data, test_data = ANN.data_preprocessing((X_train, y_train), (X_test, y_test))

#train the model and plot the stats
model.train_network(training_data, 24, 50, 3,test_data)

#save the model
filename = 'trained_model.sav'
pickle.dump(model, open(filename, 'wb'))