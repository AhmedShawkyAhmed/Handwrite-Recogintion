import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# training the model with inputs
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
# Heddin Leyars
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
# output Layes
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)

loss, accuracy = model.evaluate(x_test, y_test)

print(accuracy)
print(loss)

# save epoch to model
model.save('digital.model')

# loop in images
for x in range(1,7):
    # image proparites 
    img = cv.imread(f'{x}.png')[:,:,0]
    img = np.invert(np.array([img]))
    # compare image with the training set and print the result
    prediction = model.predict(img)
    print(f'the result is probablt: {np.argmax(prediction)}')
    # display image in frame with white background
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()