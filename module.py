import random
import numpy as np 
import pandas 
import matplotlib.pyplot as plt 
from copy import deepcopy as c 
import os


def SynthesizeData(args):
    p, d, n, r = args.p, args.d, args.n, args.r
    fill_num = int(p*d*n)
    U_star = np.random.randn(d,r)
    V_star = np.random.randn(n,r)
    S_star = np.zeros(d*n)
    fill = np.random.normal(0,10,fill_num)
    loc = np.arange(d*n)
    random.shuffle(loc)
    S_star[loc[:fill_num]] = fill
    X = U_star@V_star.T + S_star.reshape(d,n)
    U0 = np.random.randn(d,r)
    V0 = np.random.randn(n,r)
    return X,U_star@V_star.T,U0,V0



class MF_Model(object):
    def __init__(self,args,X):
        self.X = X
        self.model_name = args.model_name

    def __call__(self,U,V,y,target_grad = 'u'):
        if self.model_name == 'subgradient':
            return self.subgradient(U,V,target = target_grad)
        if self.model_name == 'A_IRLS_combined':
            coef = U if target_grad == 'v' else V
            return self.A_IRLS_combined(coef,target = target_grad)
        if self.model_name == 'A_IRLS':
            coef = U if target_grad == 'v' else V
            return self.A_IRLS(y,coef)
    
    def subgradient(self,U,V,target = 'u'):
        X = self.X
        res = X-U@V.T
        t = np.ones_like(X)
        t[res < 0] = -1
        t[res == 0] = np.random.uniform(-1+1e-6,1)
        if target == 'v':
            return t.T@U 
        else:
            return t@V

    def A_IRLS_combined(self, coef, cri = 0.01, max_iter = 150,delta = 1e-5,target = 'v'):
        X = self.X
        d,n = X.shape
        _,r = coef.shape
        if target == 'v':
            X_vec = X.transpose(1,0).reshape(-1)
            coef_expand = np.zeros((d*n,r*n))
            for i in range(n):
                coef_expand[d*i:d*(i+1),r*i:r*(i+1)] = coef
        if target == 'u':
            X_vec = X.reshape(-1)
            coef_expand = np.zeros((n*d,r*d))
            for i in range(d):
                coef_expand[n*i:n*(i+1),r*i:r*(i+1)] = coef
        W0 = np.eye(d*n)
        diff = float('inf')
        for _ in range(max_iter):
            if diff <= cri:
                break
            beta = np.linalg.inv(coef_expand.T@W0@coef_expand)@coef_expand.T@W0@X_vec
            tr = np.abs(X_vec-coef_expand@beta)
            tr[np.where(tr < delta)] = delta
            W1 = np.diag(1/tr)
            diff = np.abs(np.sum(W1-W0))/(d*n)
            W0 = c(W1)
        return beta.reshape(d,r) if target == 'u' else beta.reshape(n,r)

    def A_IRLS(self, y, coef, cri = 0.01, max_iter = 150, delta = 1e-5):
        n,_ = coef.shape
        w = np.eye(n)
        diff = float('inf')
        for _ in range(max_iter):
            if diff <= cri:
                break
            else:
                beta = np.linalg.inv(coef.T@w@coef)@coef.T@w@y
                tr = np.abs(y-coef@beta)
                tr[np.where(tr < delta)] = delta
                w_new = np.diag(1/tr)
                diff = np.abs(np.sum(w_new-w))
                w = c(w_new)
        return beta

def plot_error(err_ls, args):
    model_name, save_dir = args.model_name, args.save_dir
    plt.figure(figsize=(10,8))
    plt.plot(np.arange(len(err_ls)),err_ls,LineWidth = 1)
    plt.title(f'Error Curve with {model_name} algorithm')
    plt.xlabel(r'#iter')
    plt.ylabel(r'$\|\|UV^T-L^{\star}\|\|$')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    path = os.path.join(save_dir,f'{model_name}.png')
    plt.savefig(path)
    plt.show()

    
