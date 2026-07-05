import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import transforms
from torch.optim import Adam

#https://www.digitalocean.com/community/tutorials/writing-cnns-from-scratch-in-pytorch

a = torch.tensor([1,2,3],requires_grad=False)
b = torch.rand(1,3,requires_grad=False)
d = torch.ones(3,4) # 3 rows, 4 cols

c = a*b

# AutoGrad: Automatic Differentiation (VERY IMPORTANT)
x = torch.tensor(2.0,requires_grad=True)
y = torch.tensor(3.0,requires_grad=True)

z = x**3+y**3
z.backward() # this computes dz/dx(which is 12) and dz/dy.(which is 27). (this is how partial deravaties calculated for loss function as a function of n number of weights.

# CNN using pytorch.
class Network(nn.Module):
    def __init__(self, input_shape,hid_layers, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_shape,hid_layers)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hid_layers,num_classes)

    def forward(self,x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X,y = iris.data,iris.target # 4 values to check # finally 3 classes

X_train,X_test, y_train,y_test = train_test_split(X,y,random_state=42,test_size=0.2)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(len(X_train))
print(len(X_test))
print(len(y_train))
print(len(y_test))

torch.manual_seed(42)

#define input shapes, hidden layers, output shape
input_shape = X.shape[1]
hidden_layers = 10
output_layers = len(iris.target_names)

#model object
model = Network(input_shape,hidden_layers,output_layers)

#define Loss function and optimizer
criteria = nn.CrossEntropyLoss()
optim = Adam(model.parameters(),lr=0.01)

#Convert to pytorch tesnors
y_train_tensor = torch.LongTensor(y_train)
X_train_tensor = torch.FloatTensor(X_train)

#train model
epochs = 100
for epoch in range(epochs):
    outpus = model(X_train_tensor)
    loss = criteria(outpus,y_train_tensor)

    optim.zero_grad()
    loss.backward()
    optim.step()

    # Print the loss every 10 epochs
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

#Test on test data
with torch.no_grad():
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    outputs = model(X_test_tensor)
    _,predicted = torch.max(outputs,1)
    accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
    print(f'Accuracy on the test set: {accuracy:.2f}')