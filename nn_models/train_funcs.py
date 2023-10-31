
import numpy as np
import time
from collections import deque

def train_network(
        model,              # model
        train_data: tuple,  # train data tuple
        val_data: tuple,    # validation data tuple
        seed: int,          # random seed
        n_epochs: int,      # number of epochs
        n_batch: int,       # number of elements per batch
        lambd: float,       # lambda, regularization parameter
        sigma: float,       # sigma, i.e. s.d. for input distortion
        lr: float,          # learning rate
        optimizer: str      # info on optimizer
    ) -> dict:
    
    # get data
    X_t, X_c, Y = train_data
    X_t_val, X_c_val, Y_val = val_data
    
    # init lists for loss
    train_loss_hist, val_loss_hist = [], []
    
    # init list for time
    time_hist = []
    
    # init indices for shuffling
    idxs = list(range(len(X_t)))
    
    # print info
    print('---\t Initializing training \t---\n')
    
    # get initial loss
    train_loss = model.compute_loss(X_t=X_t, X_c=X_c, Y=Y, lambd=lambd)
    val_loss = model.compute_loss(X_t=X_t_val, X_c=X_c_val, Y=Y_val, lambd=lambd)
    
    # save initial loss
    train_loss_hist.append(train_loss)
    val_loss_hist.append(val_loss)
    
    # print info
    print('EPOCH 0 - train loss: {:.2f}, val loss: {:.2f}'.format(
        train_loss,
        val_loss
    ))
    
    np.random.seed(seed)
    # iterate over epochs
    for epoch in range(1, n_epochs):
        # start clock
        start = time.time()
        
        # shuffle training data
        np.random.shuffle(idxs)
        X_t, X_c, Y = X_t[idxs], X_c[idxs], Y[idxs]
        
        # create batches
        for i in range(len(X_t) // n_batch):
            Xt_train_batch = X_t[i*n_batch:(i+1)*n_batch]
            Xc_train_batch = X_c[i*n_batch:(i+1)*n_batch]
            Y_train_batch = Y[i*n_batch:(i+1)*n_batch]
                
            model.train(
                X_t=Xt_train_batch,
                X_c=Xc_train_batch,
                Y=Y_train_batch,
                sigma=sigma,
                lambd=lambd,
                eta=lr,
                t=epoch
            )
        
        # end clock and save time
        train_time = time.time() - start
        time_hist.append(train_time)
        
        # get loss
        train_loss = model.compute_loss(X_t=X_t, X_c=X_c, Y=Y, lambd=lambd)
        val_loss = model.compute_loss(X_t=X_t_val, X_c=X_c_val, Y=Y_val, lambd=lambd)
        
        # print info
        print('EPOCH {} ({:.2f} s) - train loss: {:.2f}, val loss: {:.2f}'.format(
            epoch,
            train_time,
            train_loss,
            val_loss
        ))
        
        # save loss
        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)

    hyperparams = {
        'n_epochs':n_epochs,
        'n_batch':n_batch,
        'lambd':lambd,
        'sigma':sigma,
        'lr':lr,
        'optimizer':optimizer
    }

    # package output
    results = {
        'train_loss':train_loss_hist, 
        'val_loss':val_loss_hist,
        'train_time':time_hist,
        'weights':model.weights,
        'hyperparams':hyperparams
    }
    
    return results

def test_autoreg(
        model,
        X_t: np.array,
        X_c: np.array,
        Y: np.array,
        t_steps: int
    ):
    
    preds = []
    encodings = []
    
    pred_queue = deque(maxlen=t_steps)
    for y in Y[:t_steps]:
        pred_queue.append(y[np.newaxis, :])
        preds.append(y)
        
    for (x_t, x_c) in zip(X_t, X_c):
        y = pred_queue.popleft()
        y_pred, encoded = model.predict(X_t=x_t[np.newaxis, :], X_c=x_c[np.newaxis, :], train=False)
        pred_queue.append(y_pred)
        preds.append(y_pred[0])
        encodings.append(encoded[0])
        
    return preds, encodings