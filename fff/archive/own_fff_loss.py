def my_fff_loss(model, input, beta=1., mmd_factor=0.):
  v = torch.randn_like(input, requires_grad=True)
  z, v1 = torch.autograd.functional.vjp(model.forward_net, input, v, create_graph=True)
  x_rec, v2 = torch.autograd.functional.jvp(model.backward_net, z, v, create_graph=True)
  nogradprod = torch.squeeze(torch.bmm(v2.detach().unsqueeze(-2), v1.unsqueeze(-1)))
  ml_loss = 0.5*torch.sum(z**2, dim=-1) - nogradprod
  rec_loss = torch.sum((input - x_rec)**2, dim=-1)
  if mmd_factor != 0.:
    z = model(input)
    mmd = mmd_inverse_multi_quadratic(z, torch.randn_like(z))
    return ml_loss + beta*rec_loss + mmd_factor*mmd
  return ml_loss + beta*rec_loss

def mmd_inverse_multi_quadratic(x, y, bandwidths=None):
  batch_size = x.size()[0]
  # compute the kernel matrices for each combination of x, y
  # (cleverly using broadcasting to do this efficiently)
  xx, yy, xy = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x,y.t())
  rx = (xx.diag().unsqueeze(0).expand_as(xx))
  ry = (yy.diag().unsqueeze(0).expand_as(yy))

  # compute the sum of kernels at different bandwidths
  K, L, P = 0, 0, 0
  if bandwidths is None:
    bandwidths = [0.4, 0.8, 1.6]
  for sigma in bandwidths:
    s = 1.0 / sigma**2
    K += 1.0 / (1.0 + s * (rx.t() + rx - 2.0*xx))
    L += 1.0 / (1.0 + s * (ry.t() + ry - 2.0*yy))
    P += 1.0 / (1.0 + s * (rx.t() + ry - 2.0*xy))

  beta = 1./(batch_size*(batch_size-1)*len(bandwidths))
  gamma = 2./(batch_size**2 * len(bandwidths))
  return beta * (torch.sum(K)+torch.sum(L)) - gamma * torch.sum(P)
