import torch
import torch.nn as nn
import torch.optim as optim

input = torch.tensor([[0,0],[0,1],[1,0],[1,1]],dtype=torch.float32)
labels = torch.tensor([[0],[1],[1],[0]],dtype=torch.float32)

epochs = 2500
learning_rate = 0.001

class XOR(nn.Module):
    def __init__(self):
        super(XOR,self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_features=2,out_features=8),
                                    nn.Sigmoid())
        self.layer2 = nn.Sequential(nn.Linear(in_features=8,out_features=1),
                                    nn.Sigmoid())
        
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


model = XOR()
costfn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    output = model(input)
    loss = costfn(output,labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 500 == 0 or epoch == 0:
        print(f"Epoch {epoch + 1} loss {loss.item():.6f}")


model.eval()
with torch.no_grad():
    test_outputs = model(input)
    predicted_class = torch.round(test_outputs) 
    
    for i in range(len(input)):
        print(f"Input: {input[i].tolist()} -> Predicted Probability: {test_outputs[i].item():.4f} -> Gate Output: {int(predicted_class[i].item())}")