#Autoencoder block
def auto_encoder(input_layer):
  #Encoder
  encode = Conv1D(filters=96, kernel_size=3, strides=1, padding='same')(input_layer)
  encode = ReLU()(encode)
  encode = MaxPooling1D(pool_size=2, strides=1, padding='same')(encode)
  encode = Conv1D(filters=16, kernel_size=3, strides=1, padding='same')(encode)
  encode = ReLU()(encode)
  encode = MaxPooling1D(pool_size=2, strides=1, padding='same')(encode)
  encode = Conv1D(filters=32, kernel_size=3, strides=1, padding='same')(encode)
  encode = ReLU()(encode)
  encode = MaxPooling1D(pool_size=2, strides=1, padding='same')(encode)
  return encode