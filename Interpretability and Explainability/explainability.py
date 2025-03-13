from keras.models import Model
import matplotlib.pyplot as plt
#Explainability
model_layer = model.layers[::-1]
#Select convolution layer
print(model_layer[6])
# Create an intermediate model to get the output of the current layer
intermediate_model = Model(inputs=model.input, outputs=model_layer[6].output)
# Predict using the intermediate model to get the output of the layer
layer_output_stressed = intermediate_model.predict(x_test_stressed) #input the x_test_normal for normal class
# Ensure the output is a valid shape for time series (batch_size, time_steps, filters)
num_filters = layer_output_stressed.shape[-1]
time_steps = layer_output_stressed.shape[1]
# Visualize the output of each filter
for i in range(num_filters):
    print(f'Filter-{i}')
    plt.figure(figsize=(1,1))
    plt.plot(layer_output_stressed[0, 5:, i], color='darkmagenta', linewidth=2)
    plt.axis('off')
    plt.show()