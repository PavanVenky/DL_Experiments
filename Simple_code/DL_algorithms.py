import torch

def Categorical_cross_entropy():
    softmax_out = torch.tensor([0.1,0.5,0.8,0.9,0.99]) # consider car as GT. model predicted outputs as softmax outputs. (probability 0.1-->very wrong..)

    #cross entropy
    CE = -torch.log(softmax_out)
    print(CE)                                                   # CE loss = -(sum(yi*log(pi)))

    gamma = [1,2,3,4,5]

    for idx,gamma in enumerate([1,2,3,4,5]):
        focal_loss = (1-softmax_out[idx])**gamma * CE[idx]      # Focal loss = CE * (1-pi)**gamma  here (1-pi)**gamma is focal factor
        print(focal_loss)

    #Single line for focal loss: Focal Loss does NOT suppress hard/wrong examples very much. It mainly suppresses easy/correct examples.
    # https://chatgpt.com/share/6a8c767c-9028-83e8-9fb1-34750b8103ac


Categorical_cross_entropy()