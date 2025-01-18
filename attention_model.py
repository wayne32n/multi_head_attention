from keras import layers, Model
import numpy as np

# Define input shape
input_shape = (4, 4, 64)

# Create a simple model
def create_attention_model():
    # Input layer
    inputs = layers.Input(shape=input_shape)
    
    # MultiHeadAttention layer
    attention_layer = layers.MultiHeadAttention(
        num_heads=2,
        key_dim=64,
        value_dim=64
    )
    attention_output = attention_layer(inputs, inputs)
    
    # Create model
    model = Model(inputs=inputs, outputs=attention_output)
    return model, attention_layer

# Create and compile the model
model, attention_layer = create_attention_model()
model.compile(optimizer='adam', loss='mse')

# Print model summary
model.summary()

# Example: Generate some random data for testing
x_train = np.random.random((10, 4, 4, 64))  # 10 samples
y_train = np.random.random((10, 4, 4, 64))  # 10 samples

# Train the model
history = model.fit(x_train, y_train, epochs=1, batch_size=2)

# Print the trained parameters
print("\nTrained Parameters:")
for weight in attention_layer.weights:
    print(f"\n{weight.name}:")
    print(f"Shape: {weight.shape}")
    print(f"Values:\n{weight.numpy()}")

# Print training history
print("\nTraining History:")
for epoch, loss in enumerate(history.history['loss']):
    print(f"Epoch {epoch + 1}: Loss = {loss:.4f}") 

# After training, save the weights
trained_weights = [weight.numpy() for weight in attention_layer.weights]
np.save('attention_weights.npy', trained_weights) 