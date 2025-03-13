#Convolution block
def conv_block(input_layer, filters):
  f2, f3, f4 = filters
  #path1
  path1 = Conv1D(filters=f4, kernel_size=1, strides=1, padding='same')(input_layer)
  path1 = ReLU()(path1)
  #path2
  path2 = Conv1D(filters=f2, kernel_size=9, strides=1, padding='same')(input_layer)
  path2 = ReLU()(path2)
  path2 = Conv1D(filters=f4, kernel_size=3, strides=1, padding='same')(path2)
  path2 = ReLU()(path2)
  #path3
  path3 = Conv1D(filters=f3, kernel_size=3, strides=1, padding='same')(input_layer)
  path3 = ReLU()(path3)
  path3 = Conv1D(filters=f4, kernel_size=9, strides=1, padding='same')(path3)
  path3 = ReLU()(path3)
  #Concatenate
  x = Concatenate(axis=-1)([path1, path2, path3])
  return x