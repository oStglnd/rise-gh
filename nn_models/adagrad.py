
import numpy as np

class AdaGrad:
    def __init__(
            self,
            eps: float,
            weights: list
        ):
        # save init params
        self.eps = eps
        
        # init dicts for saving moments
        self.m = {}
        
        # init moments
        for name, weight in weights.items():
            self.m[name] = np.zeros(weight.shape)
    
    def step(
            self, 
            weight: str, 
            grad: np.array
        ) -> np.array:
        
        self.m[weight] += np.square(grad)
        step_update = grad / np.sqrt(self.m[weight] + self.eps)
        
        return step_update
