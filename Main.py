import numpy as np
import os
import glob
import cv2


def prepare_training_data():
    trainImgs = []
    trainLbls = []
    testImgs = []
    testLbls = []

    img_list = sorted(glob.glob('Dataset/*.BMP'))
    print(len(img_list))
    test_list = sorted(glob.glob('Test/*.BMP'))
    print(len(test_list))

    for i, img_path in enumerate(img_list):
        img = cv2.imread(img_path)
        img = binarizate(img)
        img = cv2.resize(img, (64, 64))
        trainImgs.append(img)
        trainLbls.append(extract_label(img_path))


    for i, img_path in enumerate(test_list):
        img = cv2.imread(img_path)
        img = binarizate(img)
        img = cv2.resize(img, (64, 64))
        testImgs.append(img)
        testLbls.append(extract_label(img_path))

    TrainX = np.array(trainImgs)
    TrainY = np.array(trainLbls)
    TestX = np.array(testImgs)
    TestY = np.array(testLbls)
    return TrainX, TrainY, TestX, TestY


def extract_label(img_path):
    filename, _ = os.path.splitext(os.path.basename(img_path))
    subject_id, rest = filename.split('__')
    gender, rest1, rest2, _ = rest.split('_')
    gender = 0 if gender == 'M' else 1
    # assigned "1" value to females so feminists wont rage
    return gender



def binarizate(gray):
    pixel_number = gray.shape[0] * gray.shape[1]
    mean_weigth = 1.0/pixel_number
    his, bins = np.histogram(gray, np.arange(0, 257))
    final_thresh = -1
    final_value = -1
    intensity_arr = np.arange(256)
    for t in bins[1:-1]:
        pcb = np.sum(his[:t])
        pcf = np.sum(his[t:])
        Wb = pcb * mean_weigth
        Wf = pcf * mean_weigth

        mub = np.sum(intensity_arr[:t]*his[:t]) / float(pcb)
        muf = np.sum(intensity_arr[t:]*his[t:]) / float(pcf)
        value = Wb * Wf * (mub - muf) ** 2

        if value > final_value:
            final_thresh = t
            final_value = value
    final_img = gray.copy()
    final_img[gray > final_thresh] = 255
    final_img[gray < final_thresh] = 0
    return final_img


# logistic function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))



# gradient descent with regularisation
# Maximum-likelihood estimation is a common learning algorithm for Logistic Regression but implementing Gradient Descent is much simpler. So here we go.
def gradient_descent(w, b, X, y, learning_rate=0.005, lmd=10, no_of_iteration=2000):
    m = X.shape[0]
    for i in range(no_of_iteration):

        z = np.matmul(X, w) + b
        hx = sigmoid(z)

        dw = (1 / m) * np.matmul(X.T, hx - y)
        db = (1 / m) * np.sum(hx - y)

        factor = 1 - ((learning_rate * lmd) / m)

        w = w * factor - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            print("Iteration {}...".format(i))

    return w, b


# plugging numbers into the logistic regression equation and calculating the result. If its < 0.5 its a male(0). Otherwise its a female(1).
def predict(w, b, X):
    m = X.shape[0]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[1], 1)
    A = sigmoid(np.dot(w.T, X.T) + b)
    for i in range(A.shape[1]):
        if A[0, i] <= 0.5:
            Y_prediction[0, i] = 0
        else:
            Y_prediction[0, i] = 1

    return Y_prediction

trainImgs, trainLbls, testImgs, testLbls = prepare_training_data()



# The original shape: (N,px,px,3)
# After reshaping : (N,(px)x(px)x3)
TrainSet = trainImgs.reshape(trainImgs.shape[0], -1)
TestSet = testImgs.reshape(testImgs.shape[0], -1)

# Performing scaling so that convergence during gradient descent happens faster than usual
TrainSet = TrainSet/255
TestSet = TestSet/255

# Number of samples
m = TrainSet.shape[0]       # should be 5950 with current dirs
n = TestSet.shape[0]        # should be 50

# W: Weights vector
# B: Bias variable
w = np.zeros(TrainSet.shape[1], dtype=np.float64)
b = 0.0

# Performing gradient descent
w, b = gradient_descent(w, b, TrainSet, trainLbls)


testResults = predict(w, b, TestSet)
trainResults = predict(w, b, TrainSet)

print("test accuracy: {} %".format(100 - np.mean(np.abs(testResults - testLbls)) * 100))
print("train accuracy: {} %".format(100 - np.mean(np.abs(trainResults - trainLbls)) * 100))

