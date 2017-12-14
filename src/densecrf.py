import numpy as np
import pairwise
import cv2
from compatibility import PottsCompatibility
import math
import scipy.stats as stats
class DenseCRF:

    def __init__(self, height, width, num_labels):
        self.height = height
        self.width = width
        self.num_pixels = self.height * self.width
        self.num_labels = num_labels
        self.Q = np.zeros([self.num_labels, self.width * self.height])
        self.unary = np.zeros([self.num_labels, self.width * self.height])
        self.pairwise = []
        self.n_iter = 1000;
        return

    def set_unary(self, unary):
        self.unary = unary
    
    def add_pairwise_energy(self, potential):
        self.pairwise.append(potential)

    def add_pairwise_gaussian(self, sx, sy, function, kernel_type=pairwise.KernelType.DIAG_KERNEL, normalization_type=pairwise.NormType.NORMALIZE_SYMMETRIC):
        feature = np.zeros([2, self.num_pixels])
        for i in range(0, self.height):
            for j in range(0, self.width):
                feature[0, i * self.width + j] = j / sx;
                feature[1, i * self.width + j] = i / sy;
        self.add_pairwise_energy(pairwise.PairwisePotential(feature, function, kernel_type, normalization_type))

    def add_pairwise_bilateral(self, sx, sy, sr, sg, sb, img, function, kernel_type=pairwise.KernelType.DIAG_KERNEL, normalization_type=pairwise.NormType.NORMALIZE_SYMMETRIC):
        feature = np.zeros([5, self.num_pixels])
        for i in range(0, self.height):
            for j in range(0, self.width):
                feature[0, i * self.width + j] = j / sx;
                feature[1, i * self.width + j] = i / sy;
                feature[4, i * self.width + j] = img[i, j, 0] / sb;
                feature[3, i * self.width + j] = img[i, j, 1] / sg;
                feature[2, i * self.width + j] = img[i, j, 2] / sr;
        self.add_pairwise_energy(pairwise.PairwisePotential(feature, function, kernel_type, normalization_type))

    def kl_divergence(self, P,Q):
        P_2d = np.array(P).reshape([1,self.num_labels])
        P_3d = np.zeros([21,self.num_labels])
        P_3d[P_2d,range(self.num_labels)]=1
        vect_kl_divergence = np.vectorize(stats.entropy,otypes=[np.float],cache=False)
        return vect_kl_divergence(P_3d,Q)

    def exp_normalize(self, distrib):
        before = distrib - np.max(distrib, axis=0)
        before = np.exp(before)
        after = before / np.repeat(np.expand_dims(np.sum(before, 0), 0), self.num_labels, axis=0)
        return after

    def inference(self, n_iter):
        Q = self.exp_normalize(-self.unary)
        for i in range(0, n_iter):
            tmp = -self.unary
            for pw in self.pairwise:
                tmp = tmp - pw.apply(Q)
            Q = self.exp_normalize(tmp)
        return Q
    


