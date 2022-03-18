""" CIFAR-10 Photo Classification Dataset """

# #example of loading the cifar10 dataset
# from matplotlib import pyplot
# from keras.datasets import cifar10

# #load dataset
# (trainX, trainY), (testX, testY) = cifar10.load_data()

# #summarize loaded dataset
# # print('Train: X=%s, Y=%s' %(trainX.shape, trainY.shape))
# # print('Test: X=%s, Y=%s' % (testX.shape, testY.shape))

# #plot first few images 
# for i in range(9):
    
#     # define subplot
#     pyplot.subplot(330 + 1 + i)

#     #plot raw pixel data
#     pyplot.imshow(trainX[i])

#show the figure
# pyplot.show()

#show the testX and testY
# print(trainY)

#load train and test dataset
def load_dataset():
    
    #example of loading the cifar10 dataset
    from matplotlib import pyplot
    from keras.datasets import cifar10
    
    #load dataset
    (trainX, trainY), (testX, testY) = cifar10.load_data()
    
    #one hot encode target values
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    
    return trainX, trainY, testX, testY

# scale pixels
def prep_pixels(train,test):
    #convert from integers to floats 
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    #normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    #return normalized images
    return train_norm, test_norm

#define cnn model
def define_model():
    model = Sequential()
    # ...
    return model

#fir model 
history = 

def main():

    (trainX, trainY), (testX, testY) = load_dataset()

    print(trainX[12])

    return 1

if "__name__"=="__main__":
    main()

