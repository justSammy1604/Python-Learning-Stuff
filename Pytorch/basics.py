#Basics of Pytorch to understadn Autograd and Backpropagation
 # import numpy as np
import torch as tor  
device = tor.device('cuda:0')
X = tor.tensor([1,2,3,4,5], dtype=tor.float32)
Y = tor.tensor([2,4,6,8,10], dtype=tor.float32) 
 
w = tor.tensor(0.0, dtype=tor.float32, requires_grad=True) 
 

# model prediction
def forward(x):
    return w * x

def loss(y, y_predicted):
    return ((y_predicted - y)**2).mean()


print(f'Prediction before training: f(5) = {forward(5):.3f}')

learning_rate = 0.01
n_iters = 20

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)

    # gradients
    l.backward() # dl/dw

    # update weights
    with tor.no_grad():
        w -= learning_rate * w.grad

    w.grad.zero_()

    if epoch % 1 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')
print(f'Prediction after training: f(5) = {forward(5):.3f}')
