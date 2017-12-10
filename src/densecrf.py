import numpy as np





class DenseCRF:

    def __init__(self, height, width, num_kernels, num_labels):
        self.height = height
        self.width = width
        self.num_pixels = self.height * self.width
        self.num_kernels = num_kernels
        self.num_labels = num_labels
        self.Q = np.zeros([self.height, self.width, self.num_labels])
        self.unary = np.zeros([self.height, self.width, self.num_labels])
        return

    def exp_normlize(self, distrib):
        before = np.exp(distrib)
        after = before / np.sum(before, 2)
        return after

    def inference(self):
        
        return

    def train(self):
        return


crf = DenseCRF(10, 10, 1, 3)

Q = crf.exp_normlize(crf.unary)

print(Q)
