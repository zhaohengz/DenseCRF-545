# Author: Zhaoheng Zheng

import numpy as np

class PottsCompatibility(object):
    
    def __init__(self, weight):
        self.weight = weight

    def apply(self, Q):
        return -self.weight * Q
