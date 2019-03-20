import torch

MUL_ADDS = 1


def count_conv2d(m, x, y):
    x = x[0]

    c_in = m.in_channels
    c_out = m.out_channels
    kernel_h, kernel_w = m.kernel_size
    batch_size = x.size()[0]

    out_h = y.size(2)
    out_w = y.size(3)

    # ops per output element
    kernel_ops = MUL_ADDS * kernel_h * kernel_w * c_in // m.groups
    bias_ops = 1 if m.bias is not None else 0
    ops_per_element = kernel_ops + bias_ops

    # total ops
    output_elements = batch_size * out_w * out_h * c_out
    total_ops = output_elements * ops_per_element

    m.total_ops = torch.Tensor([int(total_ops)])


def count_convtranspose2d(m, x, y):
    x = x[0]

    c_in = m.in_channels
    c_out = m.out_channels
    kernel_h, kernel_W = m.kernel_size
    batch_size = x.size()[0]

    out_h = y.size(2)
    out_w = y.size(3)

    kernel_ops = MUL_ADDS * kernel_h * kernel_W * c_in // m.groups
    bias_ops = 1 if m.bias is not None else 0
    ops_per_element = kernel_ops + bias_ops

    ops_per_element = m.weight.nelement()
    output_elements = y.nelement()
    total_ops = output_elements * ops_per_element

    m.total_ops = torch.Tensor([int(total_ops)])


def count_bn(m, x, y):
    x = x[0]

    nelements = x.numel()
    total_ops = 4 * nelements

    m.total_ops = torch.Tensor([int(total_ops)])


def count_relu(m, x, y):
    x = x[0]

    total_ops = x.numel()

    m.total_ops = torch.Tensor([int(total_ops)])


def count_softmax(m, x, y):
    x = x[0]

    batch_size, n_features = x.size()

    total_exp = n_features
    total_add = n_features - 1
    total_div = n_features
    total_ops = batch_size * (total_exp + total_add + total_div)

    m.total_ops = torch.Tensor([int(total_ops)])


def count_maxpool(m, x, y):
    kernel_ops = torch.prod(torch.Tensor([m.kernel_size]))
    num_elements = y.numel()
    total_ops = torch.mul(kernel_ops, num_elements)

    m.total_ops = torch.Tensor([int(total_ops)])


def count_adap_maxpool(m, x, y):
    kernel = torch.div(torch.Tensor([*(x[0].shape[2:])]), torch.Tensor([(m.output_size)])).squeeze()
    kernel_ops = torch.prod(kernel)
    num_elements = y.numel()
    total_ops = torch.mul(kernel_ops, num_elements)

    m.total_ops = torch.Tensor([int(total_ops)])


def count_avgpool(m, x, y):
    total_add = torch.prod(torch.Tensor([m.kernel_size]))
    total_div = 1
    kernel_ops = torch.add(total_add, total_div)
    num_elements = y.numel()
    total_ops = torch.mul(kernel_ops, num_elements)

    m.total_ops = torch.Tensor([int(total_ops)])


def count_adap_avgpool(m, x, y):
    kernel = torch.Tensor([*(x[0].shape[2:])]) // torch.Tensor(list((m.output_size,))).squeeze()
    total_add = torch.prod(kernel)
    total_div = 1
    kernel_ops = torch.add(total_add, total_div)
    num_elements = y.numel()
    total_ops = torch.mul(kernel_ops, num_elements)

    m.total_ops = torch.Tensor([int(total_ops)])


def count_linear(m, x, y):
    total_mul = m.in_features
    total_add = m.in_features - 1
    num_elements = y.numel()
    total_ops = (total_mul + total_add) * num_elements

    m.total_ops = torch.Tensor([int(total_ops)])
