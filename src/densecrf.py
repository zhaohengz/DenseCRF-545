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
        self.n_iter = 1000;
        return

    def exp_normalize(self, distrib):
        before = np.exp(distrib)
        after = before / np.repeat(np.expand_dims(np.sum(before, 2), -1), self.num_labels, axis=2)
        return after

    def inference(self):
        Q = exp_normalize(-self.unary)
        for i in range(0, self.n_iter):
                        

    def train(self):
        return


