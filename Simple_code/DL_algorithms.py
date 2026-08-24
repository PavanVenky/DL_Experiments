import torch

def Categorical_cross_entropy():
    softmax_out = torch.tensor([0.1,0.5,0.8,0.9,0.99])

    #cross entropy
    CE = -torch.log(softmax_out)
    print(CE)                                                   # CE loss = -(sum(yi*log(pi)))

    gamma = [1,2,3,4,5]

    for idx,gamma in enumerate([1,2,3,4,5]):
        focal_loss = (1-softmax_out[idx])**gamma * CE[idx]      # Focal loss = CE * (1-pi)**gamma  here (1-pi)**gamma is focal factor
        print(focal_loss)


Categorical_cross_entropy()