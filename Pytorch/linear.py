""" """ 
""" 1. Design Model (input, output size, forward pass)
2. Construct loss and optimizer
3. Training loop
    - forward pass: compute prediction
    - backward pass: gradients
    - update weights  """ 


# Linear regression
import torch 
import torch.nn as nn
import numpy as np
from sklearn import datasets 
import matplotlib.pyplot as plt

# 0 prepare data

X_numpy,Y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

X = torch.from_numpy(X_numpy.astype(np.float32))
Y = torch.from_numpy(Y_numpy.astype(np.float32))

Y = Y.view(Y.shape[0], 1)

n_samples, n_features = X.shape


# print(n_samples, n_features)

# 1. Model
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)

# 2. Loss and optimizer
criterion = nn.MSELoss()
lr = 0.01
optim = torch.optim.SGD(model.parameters(), lr=lr)
# 3. Training loop

num_epochs = 200
for epoch in range(num_epochs):
    # forward pass and loss
    y_predicted = model(X)
    loss = criterion(y_predicted, Y)

    # backward pass
    loss.backward()

    # update
    optim.step()

    # zero gradients
    optim.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item()}')

# plot
predicted = model(X).detach().numpy()
plt.plot(X_numpy, Y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show() 
