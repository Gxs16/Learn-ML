#%%
import numpy as np
import torch

def f(x):  # Objective function
    return x ** 2

def f_grad(x):  # Gradient (derivative) of the objective function
    return 2 * x

#%%
def gd(eta, f_grad, f, epoch=10):
    x = 10.0
    result = [x]
    for i in range(epoch):
        x -= eta*f_grad(x)
        result.append(x)
        print('epoch {epoch}, x: {x:.5f}, f(x): {value:.5f}'.format(epoch=i, x=x, value=f(x)))
    return result

#result = gd(0.4, f_grad, f)
result = gd(0.6, f_grad, f)
# %%
c = torch.tensor(0.5)

def f(x):  # Objective function
    return torch.cosh(c * x)

def f_grad(x):  # Gradient of the objective function
    return c * torch.sinh(c * x)

def f_hess(x):  # Hessian of the objective function
    return c**2 * torch.cosh(c * x)

def newton(eta=1, f_hess=f_hess, f_grad=f_grad, f=f, epoch=10):
    x = 10.0
    result = []
    for i in range(epoch):
        x -= eta*f_grad(x)/f_hess(x)
        result.append(x)
        print('epoch {epoch}, x: {x:.5f}, f(x): {value:.5f}'.format(epoch=i, x=x, value=f(x)))
    return result

result = newton()
result = gd(0.4, f_grad, f, epoch=10)
# %%
