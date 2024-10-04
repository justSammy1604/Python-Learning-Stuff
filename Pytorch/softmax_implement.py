import torch
import torch.nn as nn
import numpy as np

# Softmax function

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

x = np.array([2.0, 1.5, 0.8])

output = softmax(x)
print(f'Softmax numpy: {output}')


x = torch.tensor([2.0, 1.5, 0.8])
output = torch.softmax(x, dim=0)

print(f'Softmax torch: {output}') 

# CrossEntropyLoss

loss = nn.CrossEntropyLoss()

Y = torch.tensor([2,0,1])

Y_pred_good = torch.tensor([[0.1, 1.0, 0.3], [2.0, 1.0, 0.1], [0.1, 3.0, 2.0]])
Y_pred_bad = torch.tensor([[0.5, 2.0, 0.3], [2.0, 0.5, 0.3], [0.3, 2.0, 0.5]])

l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)

print(f'Pytorch Loss1: {l1.item()} Loss2: {l2.item()}')

_, predictions1 = torch.max(Y_pred_good, 1)
_, predictions2 = torch.max(Y_pred_bad, 1)

print(f'Pytorch Predictions1: {predictions1} Predictions2: {predictions2}')
