#DSB block
def down_sampling_block(input_layer):
  x = Conv1D(filters=128, kernel_size=5, strides=4, padding='same')(input_layer)
  x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)
  x = BatchNormalization()(x)
  x_in = ReLU()(x)
  return x_in