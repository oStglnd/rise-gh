

import numpy as np




class recurrentNeuralNetwork:
    def __init__(
            self,
            m: int,
            k1: int,
            k2: int,
            T: int,
            seed: int,
            optimizer: str
            ):
        
        # init weight dims
        self.m = m     # dimensionality of recurrent layer
        self.k1 = k1   # number of recurrent inputs
        self.k2 = k2   # number of non-recurrent inputs
        self.T = T     # length of sequence
        
        # set seed
        self.seed = seed
        np.random.seed(seed)
        
        # init weight dict
        self.weights = {}
        
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
        
        # self.weights['C'][:, -self.k2:] = 1
        
        # # init bias for concatenation layer
        # self.weights['c'] = np.zeros(
        #     shape=(1, 1)
        # )
        
        # initialize hprev
        self.hprev = np.zeros(shape=(self.m, 1))
        
        # init optimizer
        self.optimizer = None
        
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
        
        Y = S @ self.weights['C'].T #+ self.weights['c']
        
        
        # if train:
        #     X_c += np.random.normal(
        #                 loc=0, scale=sigma,
        #                 size=X_c.shape
        #             )
    
    
        # S = self.weights['C'] @ h_list[-1]# + self.weights['c']
        # Y = X_c + S.T
        
        
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
        # G = 2 * (Y_hat - Y).T @ S * (1 - S)
        G = 2 * (Y_hat - Y)
        
        # # compute grads for concatenation/output layer
        C_grads = np.mean(G * S, axis=0)  + 2 * lambd * self.weights['C']
        # C_grads[:, -self.k2:] = 0
        # C_grads = np.mean(h_list[-1] * G.T, axis=1) + 2 * lambd * self.weights['C']
        # c_grads = np.mean(G, axis=0)
        
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
            # 'c':c_grads
        }
        
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
        Parameters
        ----------
        X : Nxd data matrix
        Y : NxK one-hot encoded label matrix
        lambd: regularization parameter
        eps: epsilon for incremental derivative calc.
        
        Returns
        -------
        W_gradsNum : numerically calculated gradients for weight martix (W)
        b_gradsNum : numerically calculated gradients for bias matrix (b)
        """

        # save initial weights
        gradsDict = {}

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
            gradsDict[name] = w_gradsNum
            
            # reset weigth vector
            self.weights[name] = w_0
            
        return gradsDict
    
    def compute_loss(
            self,
            X_t: np.array,
            X_c: np.array,
            Y: np.array,
            lambd: float
            ) -> float:
        
        Y_hat, _ = self.predict(
            X_t=X_t, 
            X_c=X_c, 
            train=False
        )
        
        loss = np.square(Y - Y_hat).mean()
        
        for key, weights in self.weights.items():
            # if key.isupper():
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
            
            
class feeedForwardNeuralNetwork:
    def __init__(
            self, 
            k1: int,
            k2: int,
            m: list,
            initialization: str,
            optimizer: str,
            seed: int
        ):
        
        # init seed
        np.random.seed(seed)
        
        # init weight dims
        self.k1 = k1
        self.k2 = k2
        self.m = m
        self.n_layers = len(m)
        
        # init weight dims list
        weight_list = [k1] + m
        self.layers = []
        
        # init weight dict
        self.weights = {}
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
            loc=0, scale=np.sqrt(2/(self.m[-1]+self.k2)),
            size=(1, self.m[-1]+self.k2)
        )
            
        
        # for m1, m2 in zip(weightList[:-1], weightList[1:]):
        #     layer = {}
            
        #     scale = np.sqrt(2/m1)
        #     layer['W'] = np.random.normal(
        #             loc=0, 
        #             scale=scale, 
        #             size=(m2, m1)
        #     )
            
        #     layer['b'] = np.zeros(shape=(m2, 1))
            
        #     self.layers.append(layer)
        
        
        # self.weights['C'] = 
        
        # init optimizer
        self.optimizer = None
     
    def predict(
            self, 
            X_t: np.array,
            X_c: np.array,
            train: bool,
            sigma: float = None,
        ) -> np.array:
        """
        Parameters
        ----------
        X : Nxd data matrix

        Returns
        -------
        P : KxN score matrix w. softmax activation
        """
        h_list = [X_t.T.copy()]
        for l in range(self.n_layers):
            s = self.weights['W'+str(l)] @ h_list[-1] + self.weights['b'+str(l)]
            h = np.maximum(0, s)
            h_list.append(h)
        
        # s = self.weights['W'+str(l+1)] @ h_list[-1] + self.weights['b'+str(l+1)]
        # h_list.append(s)
        
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
        
    def compute_loss(
            self,
            X_t: np.array,
            X_c: np.array,
            Y: np.array,
            lambd: float
            ) -> float:
        
        Y_hat, _ = self.predict(
            X_t=X_t,
            X_c=X_c,
            train=False
        )
        
        loss = np.square(Y - Y_hat).mean()     
        
        for key, weights in self.weights.items():
            # if key.isupper():
            loss += lambd * np.sum(np.square(weights))
            
        return loss
    
    def compute_grads(
            self, 
            X_t: np.array,
            X_c: np.array, 
            Y: np.array, 
            sigma: float,
            lambd: float
        ) -> (np.array, np.array):
        """
        Parameters
        ----------
        X : Nxd data matrix
        Y : NxK one-hot encoded label matrix
        lambd: regularization parameter
        
        Returns
        -------
        W_grads : gradients for weight martix (W)
        b_grads : gradients for bias matrix (b)
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
        
            
        # for layer, h in zip(self.layers[::-1], hList[::-1]):
        #     W_grads = N**-1 * G @ h.T + 2 * lambd * layer['W']
        #     b_grads = N**-1 * np.sum(G, axis=1)
        #     b_grads = np.expand_dims(b_grads, axis=1)
            
        #     # save grads
        #     grads_list.append({
        #         'W':W_grads,
        #         'b':b_grads
        #     })
            
        #     # propagate g
        #     h[h > 0] = 1
        #     G = layer['W'].T @ G * h
        
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
        Parameters
        ----------
        X : Nxd data matrix
        Y : NxK one-hot encoded label matrix
        lambd: regularization parameter
        eps: epsilon for incremental derivative calc.
        
        Returns
        -------
        W_gradsNum : numerically calculated gradients for weight martix (W)
        b_gradsNum : numerically calculated gradients for bias matrix (b)
        """

        # save initial weights
        gradsDict = {}

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
            gradsDict[name] = w_gradsNum
            
            # reset weigth vector
            self.weights[name] = w_0
            
        return gradsDict
    
    def train(
            self, 
            X_t: np.array, 
            X_c: np.array,
            Y: np.array, 
            sigma: float,
            lambd: float, 
            eta: float,
            t: int = None
        ):
        """
        Parameters
        ----------
        X : Nxd data matrix
        Y : NxK one-hot encoded label matrix
        lambd: regularization parameter
        eta: learning rate
        """
        # get grads from self.computeGrads and update weights
        # w. GD and learning parameter eta
        grads = self.compute_grads(X_t, X_c, Y, sigma, lambd)
        
        for key, weight in self.weights.items():
            # get update
            if t is None:
                step_update = self.optimizer.step(key, grads[key])
            else:
                step_update = self.optimizer.step(key, grads[key], t)
            
            # update weight
            weight -= eta * step_update
        