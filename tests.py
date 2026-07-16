# import torch



# tensor = torch.tensor([1,2,3,4])
# print(tensor)

# tensor = tensor.view((2, 2))
# print(tensor)


# numbers = torch.arange(start=0, end=10)
# print(numbers)


import torch
import math


def forward(timestep):
    embedding_dim = 4
    ln10000 = math.log(10000)
    vec = torch.exp(-torch.arange(embedding_dim/2)*ln10000/(embedding_dim/2 - 1))
    vec = timestep*vec
    
    sin = torch.sin(vec)
    cos = torch.cos(vec)
    
    emb = torch.cat([sin, cos])

    return emb



print(forward(1))


    