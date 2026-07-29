# Inference Engine

A lightweight Python script for running AI model inference.

## Project Structure

```text
├── inferencePy      # Core Python script for model inference
└── README.md        # Project documentation


Training vs. Inference:
  Training: The phase where a model learns from a dataset by adjusting its internal weights through backpropagation and optimization loops. It requires heavy compute power, data, and time.   

  Inference: The phase where a pre-trained model is used to make predictions or process new data on inputs it hasn't seen before. This script focuses on inference, running forward passes efficiently for local experimentation.   


How This Script Demonstrates Both:
  The Training Part (Epochs 0 to 99): It initializes weights (w_1, w_2, b) and runs a full training loop over 100 epochs. Inside the loop, it performs a forward pass to make predictions, calculates the loss (Mean Squared Error) against target values, executes a backward pass (loss.backward()) to compute gradients via backpropagation, and actively updates the weights using gradient descent.   

  The Inference Part (At the Very End): The final block (final_predictions = ...) takes the newly learned, optimized weights and performs a forward pass on the input features without computing gradients or updating parameters, using the trained model to generate final predictions.
