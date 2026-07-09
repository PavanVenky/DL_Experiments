import torch
import torch.nn as nn

input_size = 2
hidden_size = 4
output_size = 1

w1 = torch.tensor(input_size,hidden_size,requires_grad=True)
b1 = torch.tensor(hidden_size,requires_grad=True)
w2 = torch.tensor(hidden_size,output_size,requires_grad=True)
b2 = torch.tensor(output_size,requires_grad=True)

def forward(x):
    hidden = torch.matmul(w1,x) + b1
    hidden = torch.relu(hidden)

    output = torch.matmul(w2,hidden) + b2
    output = torch.relu(output)
    return output


x = torch.tensor([[1.0,2.0]])
y = forward(x)
print("Output of torch: ", y)


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN,self).__init__()
        self.hidden = nn.Linear(2,4)
        self.output = nn.Linear(4,1)

    def forward(self,x):
        x = torch.relu(self.hidden(x))
        x = self.output(x)
        return x
    
input = torch.randn([[1.0,2.0]])
model = SimpleNN()
output = model(input)
print("Output with torch.nn:", output)