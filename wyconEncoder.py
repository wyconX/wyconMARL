from keras.utils import np_utils
from keras.datasets import mnist
from keras.layers import *
from keras.models import Model
from keras.callbacks import ModelCheckpoint

import os
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def prepare_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train_r = x_train.reshape(*x_train.shape,1).astype('float32')
    x_test_r = x_test.reshape(*x_test.shape,1).astype('float32')

    x_train_n = x_train_r / 255.
    x_test_n = x_test_r / 255.

    y_train_oh = np_utils.to_categorical(y_train)
    y_test_oh = np_utils.to_categorical(y_test)

    return x_train,y_train,x_test,y_test,x_train_n,x_test_n,y_train_oh,y_test_oh

def plot_image(image):
    fig = plt.gcf()
    fig.set_size_inches(2,2)
    plt.imshow(image,cmap='binary')
    plt.show()


def plot_images_labels_prediction(images, labels,
                                  prediction, idx, num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num > 25: num = 25
    for i in range(0, num):
        ax = plt.subplot(5, 5, 1 + i)
        ax.imshow(images[idx], cmap='binary')
        title = "label=" + str(labels[idx])
        if len(prediction) > 0:
            title += ",predict=" + str(prediction[idx])

        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        idx += 1
    plt.show()

def build_model(input_width, input_height, num):

    img_input = Input(shape=(input_width, input_height,1),dtype='float32',name='image_inputs')
    #conv1
    conv1 = Conv2D(32,3,padding='same',activation='relu',kernel_initializer='he_normal')(img_input)
    conv2 = Conv2D(64,3,strides=2,padding='same', activation='relu',kernel_initializer='he_normal')(conv1)
    conv3 = Conv2D(64,3,strides=2,padding='same', activation='relu',kernel_initializer='he_normal')(conv2)
    conv4 = Conv2D(128,3,strides=2,padding='same', activation='relu',kernel_initializer='he_normal')(conv3)
    x = Flatten()(conv4)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(num, name='outputs')(x)

    model = Model(inputs=img_input, outputs=outputs)

    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    return model

def train(x,y,model):
    filepath = "modelEncoder.h5"

    checkpoint = ModelCheckpoint(
        filepath=filepath,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1,
        save_weights_only=True,
        period=3
    )
    if os.path.exists(filepath):
        model.load_weights(filepath)
    train_history = model.fit(x=x, y=y, validation_split=0.2, epochs=10, batch_size=200, callbacks=[checkpoint], verbose=2)
    return train_history

def test(model,x,y):
    filepath = "modelEncoder.h5"
    if os.path.exists(filepath):
        model.load_weights(filepath)
        print('load success')
    scores=model.evaluate(x,y)
    print(scores)

def pred(img,model):
    filepath = "modelEncoder.h5"
    if os.path.exists(filepath):
        model.load_weights(filepath)
        print('load success')
    p = model.predict_classes(img)
    print(p)

def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title("Train_history")
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


if __name__=='__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.7
    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)

    x, y, xt, yt, xn, xtn, yo, yto = prepare_data()
    _,h,w = x.shape
    n = build_model(h,w,10)
    #history = train(xn,yo,n)
    #show_train_history(history,'accuracy','val_accuracy')
    test(n, xtn, yto)

    # plot_images_labels_prediction(xt,yt,p,idx=340)

