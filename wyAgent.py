from keras.models import Model
import keras.backend as K
from keras.layers import *

class wyAgent():
    def __init__(self, action_num, alpha=0.5, gamma=0):
        self.gamma = gamma
        self.alpha = alpha
        self.action_num = action_num
        self.model = None
        self.policy = None
        self.env = None
        self.count = 0

    def build_model(self, input_height, input_width):
        img_input = Input(shape=(input_height,input_width,  1), dtype='float32', name='image_inputs')
        # conv
        conv1 = Conv2D(32, 3, padding='same', activation='relu', kernel_initializer='he_normal')(img_input)
        conv2 = Conv2D(64, 3, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(conv1)
        conv3 = Conv2D(64, 3, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(conv2)
        conv4 = Conv2D(128, 3, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(conv3)
        x = Flatten()(conv4)
        x = Dense(128, activation='relu')(x)
        output_q = Dense(self.action_num, name='output_q')(x)

        action_input = Input((self.action_num,), name='action_input')
        q_value = multiply([action_input, output_q])
        q_value = Lambda(lambda l: K.sum(l, axis=1, keepdims=True), name='q_value')(q_value)

        model = Model(inputs=[img_input, action_input], outputs=q_value)
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

        self.model = model

    def setPolicy(self, policy):
        self.policy = policy

    def setEnv(self,env):
        self.env = env





