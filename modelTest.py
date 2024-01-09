import cv2
import numpy as np
import tensorflow as tf

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    resized_image = cv2.resize(image, (188, 1))
    return resized_image.flatten() / 255.0

i = 2832
zero, one = 0, 0
model = tf.keras.models.load_model('signal_classifier.h5')

zero_test_size = 1214
one_test_size = 3154
correct = 0
incorrect = 0

while i < 4046:
    print(str(i-2831) + "/" + str(one_test_size + zero_test_size))
    test_image_path = "D:/Signal/test/\\0\data_" + str(i) + ".jpeg"
    preprocessed_image = preprocess_image(test_image_path)
    prediction = model.predict(np.array([preprocessed_image]))

    if prediction[0] >= 0.5:
        one = one + 1
    else:
        zero = zero + 1
        
    i = i + 1

str_res0 = "[ZERO TEST DATA]\nzero : " + str(zero) + "\none : " + str(one)
correct = zero
incorrect = one

i = 7352
zero, one = 0, 0

while i < 10506:
    print(str(i-7351+1214) + "/" + str(one_test_size + zero_test_size))
    test_image_path = "D:/Signal/test/\\1\data_" + str(i) + ".jpeg"
    preprocessed_image = preprocess_image(test_image_path)
    prediction = model.predict(np.array([preprocessed_image]))

    if prediction[0] >= 0.5:
        one = one + 1
    else:
        zero = zero + 1
        
    i = i + 1
    
str_res1 = "[ONE TEST DATA]\nzero : " + str(zero) + "\none : " + str(one)
correct = correct + one
incorrect = incorrect + zero
print(str_res0)
print(str_res1)
print("accuracy : (" + str(correct) + "/" + str(one_test_size + zero_test_size) + ") = " 
      + str(correct/(one_test_size + zero_test_size)))
