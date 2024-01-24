from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from core.module.unet.fp16_util import convert_module_to_f16, convert_module_to_f32
from core.module.unet.nn import (
    SiLU,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
    checkpoint,
)

# 一个壳子，它是abstract method。意思是，只要继承了这个class，你的forward一定要implement，否则runtime error。
# 这样定义，相当于标记了继承者是一个timestepblock, 且应该接受，x，emb两个输入。虽然在机制上不强制，但具有提示意义

# 在python中，一个class 必须同时使用@abstractmethod和 继承与ABC，才能被标记为abstract class。才会runtime error if not implement。
# 如果单独只是使用了一个@abstractmethod，那么仅仅是告诉读者，这个是abstactmethod，python 并不会报错

class TimestepBlock():
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        
        """
        pass

# 另一个壳子
class TimestepEmbedSequential(TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """
    def hhhh(self):
        print('test2')
        pass
    def askaskasdasdtake(self, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x




from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass

    @abstractmethod
    def perimeter(self):
        pass

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

circle = Circle(5)  # This line will raise a TypeError