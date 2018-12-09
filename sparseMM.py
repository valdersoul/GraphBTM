import torch
from torch.autograd.function import  InplaceFunction

#sparse x dense
#https://discuss.pytorch.org/t/does-pytorch-support-autograd-on-sparse-matrix/6156/5
class SparseMM(InplaceFunction):

    @staticmethod
    def forward(ctx, matrix1, matrix2):
        ctx.save_for_backward(matrix1, matrix2)
        return torch.mm(matrix1, matrix2)

    @staticmethod
    def backward(ctx, grad_output):
        matrix1, matrix2 = ctx.saved_variables
        grad_matrix1 = grad_matrix2 = None

        if ctx.needs_input_grad[0]:
            grad_matrix1 = torch.mm(grad_output, matrix2.t())

        if ctx.needs_input_grad[1]:
            grad_matrix2 = torch.mm(matrix1.t(), grad_output)

        return grad_matrix1, grad_matrix2

def sparse_ones(size):
    index = range(size)
    indices = torch.LongTensor([index, index])
    values = torch.FloatTensor(size).fill_(1)
    eye = torch.sparse.FloatTensor(indices, values, (size,size))
    return eye

def to_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())

def sparse_diag(x):
    size = x.size(0)
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)
    index = range(x.size(0))
    indices = torch.LongTensor([index, index]).cuda()
    result = sparse_tensortype(indices, x, (size,size))
    return result
