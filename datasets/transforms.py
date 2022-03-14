import torch


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., min=0, max=1.):
        self.std = std
        self.mean = mean
        self.min = min
        self.max = max

    def __call__(self, tensor):
        return torch.clamp(tensor + torch.randn(tensor.size()) * self.std + self.mean, self.min, self.max)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
