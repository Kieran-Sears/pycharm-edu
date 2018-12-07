from __future__ import print_function
import torch
import matplotlib as plt
import torch.nn as nn
import torch.functional as F

# CANNOT USE tensors of arbitrary sizes
# x_train = [[3,4,6,4], [5, 6,3,1], [3,2,5,7,4,3], [3,6,8,4,7,4,2,6,7,4,2,6]]
# length = 10
# z = np.array([xi[:length]+[None]*(length-len(xi)) for xi in x_train])
#
# # z = [[3, 5, 6], [4, 6, 5]]
# x = torch.FloatTensor(z).uniform_(-9, 9)
# print(x)

# devices used must be the same type
# t1 = torch.tensor([1,2,3])
# t2 = t1.cuda()
#
# print(t1.device)
# print(t2.device)
#
# t1 + t2

# x_train = [np.array([3,4,6,4]), np.array([5,6,3,1]), np.array([3,2,5,7,4,3]), np.array([3,6,8,4,7,4,2,6,7,4,2,6])]
# z = np.array(x_train)
#
# x = torch.tensor(x_train)
# print(x)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=5, out_channels=10, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=10, out_channels=10, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=10, out_channels=10, kernel_size=1)
        self.conv4 = nn.Conv1d(in_channels=10, out_channels=2, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.Tanh(x)
        x = self.conv2(x)
        x = nn.Tanh(x)
        x = self.conv3(x)
        x = nn.Tanh(x)
        x = self.conv4(x)
        return nn.Softmax(x)


model = Net()
model.train()

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.01)
optimizer.zero_grad()

y_ = model.forward(x)
loss = nn.ClassNLLCriterion(y_)

loss.backward(loss)
optimizer.step()

fig, ax = plt.subplots()
ax.plot(x.cpu().numpy(), y_.cpu().numpy(), ".", label="pred")
ax.plot(x.cpu().numpy(), y.cpu().numpy(), ".", label="data")
ax.set_title(f"MSE: {loss.item():0.1f}")
ax.legend()