from keras.models import Sequential, Input, Model
from keras.layers import Dense, Dropout, Conv2D, Flatten
from keras.constraints import unit_norm
from keras.optimizers import Adam

input_shape = 70*13
print(input_shape)

"""MultiLayer Perceprton implementation"""

model_dense_input = Input(shape=(input_shape,))
model_dense = Dense(units=128, activation='relu', input_dim=input_shape,
             kernel_constraint=unit_norm())(model_dense_input)
model_dense = Dropout(0.5)(model_dense)
model_dense = Dense(units=128, activation='relu')(model_dense)
model_dense = Dense(units=64, activation='relu')(model_dense)
model_dense = Dense(units=10, activation='softmax')(model_dense)
model_dense = Model(inputs=model_dense_input, output=model_dense)

model_dense.summary()

adam = Adam(lr = 0.001)
model_dense.compile(loss='categorical_crossentropy',
             optimizer = adam, metrics=['accuracy'])
model_dense.save('model_dense_untrained.h5')