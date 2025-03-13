from tensorflow.keras.layers import Conv1D, Conv1DTranspose, ReLU, Concatenate, MaxPooling1D, BatchNormalization, Dropout, GlobalAveragePooling1D, Dense, Input
from tensorflow.keras.models import Model

def TEANet_model(input_shape, num_class):
  #Input layer
  input_layer = Input(shape=input_shape)
  #Down Sampling Block call
  x_in = down_sampling_block(input_layer)
  #TEA-N, N= 2 to 7
  x_tea1 = tea_layer(x_in, [[32,96,16], [96,96,16], [64,64,16], [32,64,16], [32,32,16]])
  x_tea2 = tea_layer(x_tea1, [[96,96,16], [64,64,16], [32,32,16], [32,64,16], [96,96,16]])
  x_tea3 = tea_layer(x_tea2, [[16,32,16], [64,64,16], [64,64,32], [32,64,16], [96,96,16]])
  x_tea4 = tea_layer(x_tea3, [[32,64,16], [96,96,16], [64,64,16], [32,96,16], [16,32,16]])
  x_tea5 = tea_layer(x_tea4, [[64,64,16], [96,96,64], [64,64,96], [16,16,64], [32,32,16]])
  x_tea6 = tea_layer(x_tea5, [[64,64,16], [96,96,64], [64,64,96], [16,16,64], [32,32,16]])
  x_tea7 = tea_layer(x_tea6, [[64,64,16], [96,96,64], [64,64,96], [16,16,64], [32,32,16]])
  #Classification-1
  x_tea = Conv1D(filters=96, kernel_size=3, strides=1, padding='same')(x_tea6) #Change x_teaN, N = 2 to 7
  x_tea = ReLU()(x_tea)
  x_tea = MaxPooling1D(pool_size=2, strides=1, padding='same')(x_tea)
  x_tea = BatchNormalization()(x_tea)
  x_tea = Dropout(0.3)(x_tea)
  #Classification-2
  x_tea = Conv1D(filters=64, kernel_size=3, strides=1, padding='same')(x_tea)
  x_tea = ReLU()(x_tea)
  x_tea = MaxPooling1D(pool_size=2, strides=1, padding='same')(x_tea)
  x_tea = BatchNormalization()(x_tea)
  x_tea = Dropout(0.3)(x_tea)
  #Classification-3
  x_tea = Conv1D(filters=32, kernel_size=3, strides=1, padding='same')(x_tea)
  x_tea = ReLU()(x_tea)
  x_tea = MaxPooling1D(pool_size=2, strides=1, padding='same')(x_tea)
  x_tea = BatchNormalization()(x_tea)
  x_tea = Dropout(0.3)(x_tea)
  #Global Average Pooling Layer
  x_avgpool = GlobalAveragePooling1D()(x_tea)
  #Output prob.
  output = Dense(units=num_class, activation='softmax')(x_avgpool)
  #Model init
  model = Model(inputs=input_layer, outputs=output)
  return model