# Import torch library
import torch

# Initialize the input features (3 samples, 2 features each)
X = torch.tensor([
    [2, 3],
    [3, 4],
    [4, 5],
], dtype=torch.float32)

# Initialize the target (3 target values for the 3 samples)
target = torch.tensor([20, 30, 40], dtype=torch.float32)

# Initialize the parameters with requires_grad=True
w_1 = torch.tensor(0.5, requires_grad=True)
w_2 = torch.tensor(0.5, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

# Initialize learning rate
learningRate = 0.01

# Training loop for multiple epochs
epochs = 100

for epoch in range(epochs):
    # 1. Forward Pass: Calculate predictions
    predictions = (w_1 * X[:, 0]) + (w_2 * X[:, 1]) + b

    # 2. Loss: Compute Mean Squared Error
    loss = ((predictions - target) ** 2).mean()

    # 3. Backward Pass: Trigger backpropagation
    loss.backward()

    # 4. Update parameters and zero gradients
    with torch.no_grad():
        w_1 -= learningRate * w_1.grad
        w_2 -= learningRate * w_2.grad
        b -= learningRate * b.grad

        # Zero out gradients for the next iteration
        w_1.grad.zero_()
        w_2.grad.zero_()
        b.grad.zero_()

# Final predictions after training (detached to remove grad_fn)
final_predictions = ((w_1 * X[:, 0]) + (w_2 * X[:, 1]) + b).detach().tolist()

print("Final Predictions:")
for pred in final_predictions:
    print(f"{pred:.4f}")