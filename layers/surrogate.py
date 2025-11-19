import math
import torch

THRESH = 0.5  # neuronal threshold
LENS = 0.5  # hyper-parameters of approximate function
DECAY = 0.5  # 0.2 # decay constants

ALPHA_ERF=2.0
ALPHA_SIGMOD=4.0

def heaviside(x: torch.Tensor):
    return (x >= 0).to(x)

class pseudo_spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(THRESH).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - THRESH) < LENS
        # res= grad_input * temp.float()
        # print("temp: ",temp.float(),"restult: ",res)
        return grad_input * temp.float()

class erf(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = ALPHA_ERF
        return heaviside(x)
    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_output * (- (ctx.saved_tensors[0] * ctx.alpha).pow_(2)).exp_() * (ctx.alpha / math.sqrt(math.pi))
        return grad_x, None

class sigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = ALPHA_SIGMOD
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            sgax = (ctx.saved_tensors[0] * ctx.alpha).sigmoid_()
            grad_x = grad_output * (1. - sgax) * sgax * ctx.alpha

        return grad_x, None
