import torch

a = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(2.0, requires_grad=True)
loss = a + b
loss.backward()
print(a.grad, b.grad)
