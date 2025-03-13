import umap
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt

model_layer = model.layers[::-1] #reversing layer indices
print(model_layer[17]) #change the layer index as per model summary to select the required layer
layer = model_layer[17]
print(f"layer-{layer.name}")
# Create an intermediate model to get the output of the current layer
intermediate_model = Model(inputs=model.input, outputs=layer.output)
# Predict using the intermediate model to get the output of the layer
layer_output = intermediate_model.predict(X_test)
# Define class-color mapping
class_colors = {0: 'red', 1: 'blue'}  # Add more classes if needed
colors = [class_colors[label] for label in y_test]
# Reshape the output to 2D if necessary
layer_output_reshaped = np.reshape(layer_output, (layer_output.shape[0], -1))
#UMAP apply
umap_reducer = umap.UMAP(n_components=2, random_state=42)
layer_umap = umap_reducer.fit_transform(layer_output_reshaped)
# Plotting
plt.figure(figsize=(4, 3))
plt.scatter(layer_umap[:, 0], layer_umap[:, 1], c=colors, s=8)
# Show plot
plt.show()