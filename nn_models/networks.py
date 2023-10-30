

import numpy as np

class neuralNetwork:
    def __init__(
            self,
        ):
        
        # init optimizer
        self.optimizer = None
        
        # init weights
        self.weights = {}
    
    def predict(
            self, 
            X_t: np.array,
            X_c: np.array,
            train: bool,
            sigma: float = 0,
            ) -> np.array:
        """
        ...
        """
        return 0
        
    def compute_grads(
            self,
            X_t: np.array,
            X_c: np.array,
            Y: np.array,
            sigma: float,
            lambd: float
        ) -> dict:
        """
        ...
        """
        grads = {}
        return grads
    
    def compute_grads_numerical(
            self, 
            X_t: np.array, 
            X_c: np.array,
            Y: np.array, 
            sigma: float,
            lambd: float,
            eps: float,
        ) -> np.array:
        """
        ...
        """
        
        # save initial weights
        grads_dict = {}

        for name, weight in self.weights.items():
            shape = weight.shape
            w_perturb = np.zeros(shape)
            w_gradsNum = np.zeros(shape)
            w_0 = weight.copy()
            
            for i in range(min(shape[0], 10)):#shape[0]):
                for j in range(min(shape[1], 10)):#shape[1]):
            
                    # add perturbation
                    w_perturb[i, j] = eps
                    
                    # perturb weight vector negatively
                    # and compute cost
                    w_tmp = w_0 - w_perturb
                    self.weights[name] = w_tmp
                    loss1 = self.compute_loss(X_t, X_c, Y, lambd)
                
                    # perturb weight vector positively
                    # and compute cost
                    w_tmp = w_0 + w_perturb
                    self.weights[name] = w_tmp
                    loss2 = self.compute_loss(X_t, X_c, Y, lambd)
                    lossDiff = (loss2 - loss1) / (2 * eps)
                    
                    # get numerical grad f. W[i, j]
                    w_gradsNum[i, j] = lossDiff
                    w_perturb[i, j] = 0
        
            # save grads
            grads_dict[name] = w_gradsNum
            
            # reset weigth vector
            self.weights[name] = w_0
            
        return grads_dict
    
    def compute_loss(
            self,
            X_t: np.array,
            X_c: np.array,
            Y: np.array,
            lambd: float
            ) -> float:
        """
        ...
        """
        Y_hat, _ = self.predict(
            X_t=X_t, 
            X_c=X_c, 
            train=False
        )
        
        loss = np.square(Y - Y_hat).mean()
        
        for key, weights in self.weights.items():
            loss += lambd * np.sum(np.square(weights))
            
        return loss
    
    def train(
            self,
            X_t: np.array,
            X_c: np.array, 
            Y: np.array,
            sigma: float,
            lambd: float, 
            eta: float,
            t: int = None
        ) -> None:
        """
        ...
        """
        
        grads = self.compute_grads(
            X_t, 
            X_c, 
            Y, 
            sigma, 
            lambd
        )
        
        for key, weight in self.weights.items():
            # clip gradient
            grads[key] = np.clip(grads[key], -2, 2)
            
            # get update
            if t is None:
                step_update = self.optimizer.step(key, grads[key])
            else:
                step_update = self.optimizer.step(key, grads[key], t)
            
            # update weight
            weight -= eta * step_update
        
class recurrentNeuralNetwork(neuralNetwork):
    def __init__(
            self,
            m: int,
            k1: int,
            k2: int,
            seed: int
            ):
        
        # init super class
        super().__init__()
        
        # init weight params
        self.k1 = k1
        self.k2 = k2
        self.m = m
        
        # set seed
        self.seed = seed
        np.random.seed(seed)
        
        # init bias for recurrent layer
        self.weights['b'] = np.zeros(
            shape=(self.m, 1)
        )
        
        # init weights for recurrent layer
        self.weights['U'] = np.random.normal(
            loc=0, scale=np.sqrt(2/m), 
            size=(self.m, self.k1)
        )
        
        self.weights['W'] = np.random.normal(
            loc=0, scale=np.sqrt(2/m), 
            size=(self.m, self.m)
        )
        
        # init weight for concatenation layer
        self.weights['C'] = np.random.normal(
            loc=0, scale=np.sqrt(2/self.m),
            size=(1, self.m+self.k2)
        )
        
        # initialize hprev
        self.hprev = np.zeros(shape=(self.m, 1))
        
        
    def predict(
            self, 
            X_t: np.array,
            X_c: np.array,
            train: bool,
            sigma: float = 0,
            ) -> np.array:
        """
        Forward pass.
        
        Parameters
        ----------
        X_t : np.array
            Sequential inputs, N x T x D
        X_c : np.array
            Static inputs, N x k_2
        train : bool
            If predict for training or testing
        sigma : float, optional
            Stddev for gaussian noise

        Returns
        -------
        Y : predicted vals, N x 1

        """
        # init lists for backprop
        self.hprev = np.zeros(shape=(self.m, len(X_t)))
        h_list = [self.hprev.copy()]
        a_list = []        
        
        for x_t in X_t.transpose((1, 0, 2)):
            a = self.weights['W'] @ h_list[-1] + self.weights['U'] @ x_t.T + self.weights['b']
            h = np.tanh(a)
            
            a_list.append(a)
            h_list.append(h)
            
        if train:
            S = np.hstack((
                h_list[-1].T,
                X_c + np.random.normal(
                    loc=0, scale=sigma,
                    size=X_c.shape
                )
            ))
        else:
            S = np.hstack((
                h_list[-1].T,
                X_c
            ))
        
        Y = S @ self.weights['C'].T
        
        if train:
            self.hprev = h_list[1:]
            return a_list, h_list, S, Y
        else:
            return Y, h_list[-1].T
        
    def compute_grads(
            self,
            X_t: np.array,
            X_c: np.array,
            Y: np.array,
            sigma: float,
            lambd: float
        ) -> dict:
        
        
        # obtain predicted vals
        a_list, h_list, S, Y_hat = self.predict(
            X_t=X_t, 
            X_c=X_c, 
            train=True, 
            sigma=sigma
        )
        
        # get batch size
        N = len(X_t)
        
        # # obtain initial gradient
        G = 2 * (Y_hat - Y)
        
        # # compute grads for concatenation/output layer
        C_grads = np.mean(G * S, axis=0)  + 2 * lambd * self.weights['C']
        
        # compute remaining gradients by BPTT
        H_grad = G @ self.weights['C'][:, :self.m]
        A_grad = H_grad * (1 - np.square(np.tanh(a_list[-1].T)))
        
        A_grads = [A_grad.copy()]
        for A_t in a_list[-2::-1]:
            H_grad = A_grad @ self.weights['W']
            A_grad = H_grad * (1 - np.square(np.tanh(A_t.T)))
        
            A_grads.append(A_grad)
        
        A_grads = np.stack(A_grads[::-1]).transpose(1, 2, 0)
        H = np.stack(h_list[:-1]).transpose(2, 0, 1)
        
        U_grads = N**-1 * np.sum(A_grads @ X_t, axis=0) + 2 * lambd * self.weights['U']
        W_grads = N**-1 * np.sum(A_grads @ H, axis=0) + 2 * lambd * self.weights['W']
        b_grads = N**-1 * A_grads.sum(axis=(0, -1))[:, np.newaxis]

        grads = {
            'U':U_grads,
            'W':W_grads,
            'b':b_grads,
            'C':C_grads,
        }
        
        return grads
    
    def train(
            self,
            X_t: np.array,
            X_c: np.array, 
            Y: np.array,
            sigma: float,
            lambd: float, 
            eta: float,
            t: int = None
        ) -> None:
        
        grads = self.compute_grads(
            X_t, 
            X_c, 
            Y, 
            sigma, 
            lambd
        )
        
        for key, weight in self.weights.items():
            # clip gradient
            grads[key] = np.clip(grads[key], -5, 5)
            
            # get update
            if t is None:
                step_update = self.optimizer.step(key, grads[key])
            else:
                step_update = self.optimizer.step(key, grads[key], t)
            
            # update weight
            weight -= eta * step_update
            
            
class feedForwardNeuralNetwork(neuralNetwork):
    def __init__(
            self, 
            k1: int,
            k2: int,
            m: list,
            seed: int
        ):
        # init super class
        super().__init__()
        
        # init weight params
        self.k1 = k1
        self.k2 = k2
        self.m = m
        
        # set seed
        self.seed = seed
        np.random.seed(seed)
        
        # save n layers
        self.n_layers = len(m)
        
        # init weight dims list
        weight_list = [k1] + m
        
        # init weight dict
        for idx, (m1, m2) in enumerate(zip(weight_list[:-1], weight_list[1:])):
            scale = np.sqrt(2/m1)
            self.weights['W'+str(idx)] = np.random.normal(
                    loc=0, 
                    scale=scale, 
                    size=(m2, m1)
            )
            
            self.weights['b'+str(idx)] = np.zeros(shape=(m2, 1))

        # init weight for concatenation layer
        self.weights['C'] = np.random.normal(
            loc=0, scale=np.sqrt(2/(m[-1]+self.k2)),
            size=(1, m[-1]+self.k2)
        )
                 
    def predict(
            self, 
            X_t: np.array,
            X_c: np.array,
            train: bool,
            sigma: float = None,
        ) -> np.array:
        """
        ...
        """
        h_list = [X_t.T.copy()]
        for l in range(self.n_layers):
            s = self.weights['W'+str(l)] @ h_list[-1] + self.weights['b'+str(l)]
            h = np.maximum(0, s)
            h_list.append(h)
        
        if train:
            S = np.hstack((
                h_list[-1].T,
                X_c + np.random.normal(
                    loc=0, scale=sigma,
                    size=X_c.shape
                )
            ))
        else:
            S = np.hstack((
                h_list[-1].T,
                X_c
            ))
        
        Y = S @ self.weights['C'].T #+ self.weights['c']
        
        if not train:
            return Y, h_list[-1]
        else:
            return Y, S, h_list
    
    def compute_grads(
            self, 
            X_t: np.array,
            X_c: np.array, 
            Y: np.array, 
            sigma: float,
            lambd: float
        ) -> (np.array, np.array):
        """
        ...
        """
        # evaluate probabilities and calculate g
        Y_hat, S, h_list = self.predict(X_t, X_c, train=True, sigma=sigma)
        
        # get batch size
        N = len(Y)
        
        # # obtain initial gradient
        G = 2 * (Y_hat - Y)
        
        # init grads list
        grads = {}
        
        # get grads for concat layer
        C_grads = np.mean(G * S, axis=0)  + 2 * lambd * self.weights['C']
        grads['C'] = C_grads
        
        # propagate G
        G = (G @ self.weights['C'][:, :self.m[-1]]).T
        h = h_list[-1].copy()
        h[h>0] = 1
        G = G * h
        
        # iteratively compute grads per layer
        for l in range(self.n_layers)[::-1]:
            h = h_list[l].copy()
            
            W_grads = N**-1 * G @ h.T + 2 * lambd * self.weights['W'+str(l)]
            b_grads = N**-1 * np.sum(G, axis=1)
            b_grads = np.expand_dims(b_grads, axis=1)
            
            grads['W'+str(l)] = W_grads
            grads['b'+str(l)] = b_grads
            
            # propagate g
            
            h[h > 0] = 1
            G = self.weights['W'+str(l)].T @ G * h
        
        return grads