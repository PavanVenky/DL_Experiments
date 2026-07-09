import torch
import torch.nn as nn
import torch.optim as optim

inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
labels = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

epochs = 2000
learning_rate = 0.01

class XOR(nn.Module):
    def __init__(self):
        super(XOR, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return torch.sigmoid(self.network(x))


model = XOR()
costfn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    output = model(inputs)
    loss = costfn(output, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 500 == 0 or epoch == 0:
        print(f"Epoch {epoch + 1} loss {loss.item():.6f}")


