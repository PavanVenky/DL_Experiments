import torch

input_layers = 2
hidden_layers = 4
output_layers = 1

w1 = torch.randn(input_layers, hidden_layers, requires_grad=True)
b1 = torch.randn(hidden_layers, requires_grad=True)
w2 = torch.randn(hidden_layers, output_layers, requires_grad=True)
b2 = torch.randn(output_layers, requires_grad=True)


def mlp(x):
    hidden = torch.relu(torch.matmul(x, w1) + b1)
    output = torch.relu(torch.matmul(hidden, w2) + b2)
    return output

input_tensor = torch.tensor([[5.0, 2.7]])
output = mlp(input_tensor)
print(output)
