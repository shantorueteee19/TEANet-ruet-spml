#TEA Layer
def tea_layer(input_layer, filters):
  filters1, filters2, filters3, filters4, filters5 = filters
  #Convolution path
  path1 = conv_block(input_layer, filters1)
  path1 = BatchNormalization()(path1)
  path1 = conv_block(path1, filters2)
  path1 = BatchNormalization()(path1)
  #TEA path
  path2 = trans_conv_block(input_layer, filters3) #Transpose Convolution Block
  path2 = conv_block(path2, filters4)
  path2 = auto_encoder(path2)#Autoencoder
  path2 = auto_encoder(path2)#Autoencoder
  path2 = conv_block(path2, filters5)
  path2 = BatchNormalization()(path2)
  path2 = Conv1D(filters=path1.shape[-1], kernel_size=1, strides=1, padding='same')(path2)
  path2 = ReLU()(path2)
  path2 = MaxPooling1D(pool_size=2, strides=1, padding='same')(path2)
  #Concatenate
  x = Concatenate(axis=-1)([path1, path2])
  return x