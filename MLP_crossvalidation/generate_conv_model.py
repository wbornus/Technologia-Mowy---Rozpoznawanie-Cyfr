from keras.models import Sequential, Input, Model
from keras.layers import Dense, Dropout, Conv2D, Flatten
from keras.constraints import unit_norm
from keras.optimizers import Adam

"""implementing Convolutional Neural Network"""

model_conv_input = Input(shape=(90, 13, 1))
model_conv = Conv2D(filters=20, kernel_size=(10,10), strides=(6,6),
                    padding='same', activation='sigmoid')(model_conv_input)
# model_conv = Conv2D(filters=16, kernel_size=(5,5), strides=(4,4),
#                     padding='same', activation='sigmoid')(model_conv)
model_conv = Flatten()(model_conv)
model_conv = Dense(units=128, activation='relu')(model_conv)
model_conv = Dense(units=64, activation='relu')(model_conv)
model_conv = Dropout(0.3)(model_conv)
model_conv = Dense(units=10, activation='softmax')(model_conv)
model_conv = Model(inputs=model_conv_input, output=model_conv)

model_conv.summary()

adam = Adam(lr = 0.001)
model_conv.compile(loss='categorical_crossentropy',
              optimizer = adam, metrics=['accuracy'])

model_conv.save('model_conv_untrained.h5')