import numpy as np
import permutohedral

class NormType(object):
    NO_NORMALIZATION = 'no_norm'
    NORMALIZE_BEFORE = 'before'
    NORMALIZE_AFTER = 'after'
    NORMALIZE_SYMMETRIC = 'symmetric'

class KernelType(object):
    CONST_KERNEL = 'const'
    DIAG_KERNEL = 'diag'
    FULL_KERNEL = 'full'


class Kernel(object):
    
    def __init__(self, f, kernel_type, norm_type):
        if kernel_type == KernelType.DIAG_KERNEL:
            self.parameters = np.ones(f.shape[0])
        else:
            self.parameters = np.eye(f.shape[0])
        self.lattice = permutohedral.Permutohedral()
        self.norm_type = norm_type
        self.kernel_type = kernel_type
        self.feature = f
        self.init_lattice(self.feature)
           
    def init_lattice(self, f): 
        self.lattice.init(f)
        self.norm = self.lattice.compute(np.ones([f.shape[1], 1], dtype=np.float32).transpose()).transpose()
        
        if (self.norm_type == NormType.NO_NORMALIZATION):
            mean_norm = np.mean(self.norm)
            self.norm = np.ones(self.norm.shape) * mean_norm
        elif (self.norm_type == NormType.NORMALIZE_SYMMETRIC):
            self.norm = 1.0 / np.sqrt(self.norm + 1e-20)
        else:
            self.norm = 1.0 / (self.norm + 1e-20)

        
    def filter(self, mat_in, transpose):
        if (self.norm_type == NormType.NORMALIZE_SYMMETRIC) or (self.norm_type == NormType.NORMALIZE_BEFORE and not transpose) or (self.norm_type == NormType.NORMALIZE_AFTER and transpose):
            tmp = mat_in * np.diag(self.norm)
        else:
            tmp = mat_in

        tmp = tmp.astype(np.float32)
        if (transpose):
            self.lattice.compute(tmp, tmp, True)
        else:
            self.lattice.compute(tmp, tmp, False)
        

        if (self.norm_type == NormType.NORMALIZE_SYMMETRIC) or (self.norm_type == NormType.NORMALIZE_BEFORE and transpose) or (self.norm_type == NormType.NORMALIZE_AFTER and not transpose):
            tmp = tmp * np.diag(self.norm)

        return tmp

    def apply(self, mat_in):
        return self.filter(mat_in, False)

    def apply_transpose(self, mat_in):
        return self.filter(mat_in, True)

    def set_parameters(self, vec):
        if (self.kernel_type == KernelType.DIAG_KERNEL):
            self.parameters = vec;
            self.init_lattice(np.diag(p) * self.feature)
        elif (self.kernel_type == KernelType.FULL_KERNEL):
            tmp = np.resize(p, self.parameters.shape)
            self.parameters = tmp
            self.init_lattice(tmp * self.feature)


class PairwisePotential(object):
    
    def __init__(self, features, compatibility, kernel_type=KernelType.CONST_KERNEL, norm_type=NormType.NORMALIZE_SYMMETRIC):
        self.features = features.astype(np.float32)
        self.compatibility = compatibility
        self.kernel = Kernel(self.features, kernel_type, norm_type)
      

    def apply(self, Q):
        tmp = self.kernel.apply(Q)
        tmp = self.compatibility.apply(tmp)
        return tmp

    def apply_transpose(self, Q):
        tmp = self.kernel.apply_transpose(Q)
        tmp = self.compatibility.apply_transpose(tmp)
        return tmp


