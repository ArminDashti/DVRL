'''
Deep Variational Reinforcement Learning (2018) - DVRL
Maximilian Igl, Luisa Zintgraf, Tuan Anh Le, Frank Wood, Shimon Whiteson
https://arxiv.org/abs/1806.02426
https://github.com/maximilianigl/DVRL/blob/master/code/policy.py
'''


from torch.autograd import Variable
import collections
import torch


def resample(value, ancestral_index):
    assert(ancestral_index.size() == value.size()[:2])
    ancestral_index_unsqueezed = ancestral_index

    for _ in range(len(value.size()) - 2):
        ancestral_index_unsqueezed = \
            ancestral_index_unsqueezed.unsqueeze(-1)

    return torch.gather(value, dim=1, index=ancestral_index_unsqueezed.expand_as(value))


class State:
    def __init__(self, **kwargs):
        object.__setattr__(self, '_items', {})

        for name in kwargs:
            self._set_value_(name, kwargs[name])


    def __contains__(self, key):
        return key in self._items


    def __getattr__(self, name):
        return self._items[name]


    def __setattr__(self, name, value):
        self._set_value_(name, value)


    def __getitem__(self, key):
        if isinstance(key, str):
            return getattr(self, key)

        if key==0:
            for key, value in self._items.items():
                return value


    def __setitem__(self, name, value):
        self._set_value_(name, value)


    def __str__(self):
        return str(self._items)


    def _set_value_(self, name, value):
        self._items[name] = value
        return self


    def index_elements(self, key):
        new_state = State()
        for name, value in self._items.items():
            if isinstance(value, Variable):
                setattr(new_state, name, value[key])
            else:
                setattr(new_state, name, value.index(key))
        return new_state


    def unsequeeze_and_expand_all_(self, dim, size):
        def fn(tensor):
            dims = list(tensor.size())
            dims.insert(dim, size)
            return tensor.unsqueeze(dim).expand(*dims).contiguous()

        return self.apply_each_(fn)


    def multiply_each(self, mask, only):
        new_state = State()
        for name, value in self._items.items():
            if name not in only:
                continue
            xfactor = mask
            dims_factor = len(mask.size())
            dims_value = len(value.size())
            for i in range(dims_value - dims_factor):
                xfactor = xfactor.unsqueeze(-1)
            setattr(new_state, name, value * xfactor)
        return new_state


    def apply_each_(self, fn):
        for name, value in self._items.items():
            self._items[name] = fn(value)
        return self


    def apply_each(self, fn):
        new_state = State()
        for name, value in self._items.items():
            setattr(new_state, name, fn(value))
        return new_state


    def clone(self):
        state = State()
        for key, value in self._items.items():
            setattr(state,key,value.clone())
        return state


    def cpu_(self):
        return self.apply_each_(lambda x: x.cpu())


    def cuda_(self):
        return self.apply_each_(lambda x: x.cuda())


    def cuda(self):
        return self.apply_each(lambda x: x.cuda())


    def to(self, device):
        return self.apply_each(lambda x: x.to(device))


    def detach_(self):
        return self.apply_each_(lambda x: x.detach())


    def requires_grad_(self):
        return self.apply_each_(lambda x: x.requires_grad_())


    def detach(self):
        return self.apply_each(lambda x: x.detach())


    def resample(self, ancestral_index):
        new_state = self.clone()
        return new_state.resample_(ancestral_index)


    def resample_(self, ancestral_index):
        return self.apply_each_(lambda value: resample(value, ancestral_index))


    def to_tensor_(self):
        return self.apply_each_(lambda value: value.data)


    def to_variable_(self, **kwargs):
        return self.apply_each_(lambda value: Variable(value, **kwargs))


    def update(self, second_state):
        if second_state is not None:
            self._items.update(second_state._items)
        return self