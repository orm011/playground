import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

class LinearModel:
    def __init__(self, *,
         label_loss_type : str = 'hinge_squared_loss',
         reg_norm_lambda : float = 10.0, # equiv to C = .1 in sklearn
         regularizer_vector : np.ndarray | None = None,
         reg_vector_lambda : float = 1.0,
         class_weight : str = 'balanced',
         fit_intercept : bool = True,
         verbose : bool = False,
         max_iter : int = 10
        ):

        self.regularizer_vector = torch.from_numpy(regularizer_vector).float() if regularizer_vector is not None else None
        assert label_loss_type in ['ce_loss', 'hinge_loss', 'hinge_squared_loss']
        self.label_loss_type = label_loss_type
        self._module : nn.Linear | None = None
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.reg_norm_lambda = reg_norm_lambda
        self.reg_vector_lambda = reg_vector_lambda
        self.class_weight = class_weight

    def _step(self, batch):
        assert self._module
        X,y=batch # note y can be a floating point
        assert not y.isnan().any()
        assert not X.isnan().any()

        logits = self._module(X).reshape(-1)
        pos_total = (y == 1).float().sum()
        neg_total = X.shape[0] - pos_total
        bincount = torch.tensor([neg_total, pos_total])
        assert pos_total > 0 and neg_total > 0

        if self.class_weight == 'balanced':
                weights = y.shape[0] / (2 * bincount)
        else:
            assert False, 'unknown pos weight type'

        point_weights = weights[y.int()]

        if self.label_loss_type == 'ce_loss':
            error_loss = F.binary_cross_entropy_with_logits(logits, y, weight=None,
                                        reduction='none', pos_weight=None)
        elif self.label_loss_type in ['hinge_loss', 'hinge_squared_loss']:
            yprime = 2*y - 1 # make it 1, -1
            error_loss = torch.maximum(torch.tensor(0.),  1. - yprime * logits)
            if self.label_loss_type == 'hinge_squared_loss':
                error_loss = .5*error_loss**2
        else:
            assert False, 'unknown loss type'

        item_losses = error_loss * point_weights
        norm_squared = self._module.weight.reshape(-1) @ self._module.weight.reshape(-1)
        loss_norm = self.reg_norm_lambda * .5 * norm_squared

        if self.regularizer_vector is not None:
            loss_regularizer_vector = 1. - F.cosine_similarity(self.regularizer_vector.reshape(1,-1), self._module.weight).sum()
        else:
            loss_regularizer_vector = torch.tensor(0.)

        loss_regularizer_vector = self.reg_vector_lambda * loss_regularizer_vector
        loss_labels = item_losses.sum()
        total_loss = loss_labels + loss_norm + loss_regularizer_vector

        ans =  {
            'loss_norm' : loss_norm,
            'loss_regularizer_vector': loss_regularizer_vector,
            'loss_labels': loss_labels,
            'loss': total_loss,
        }
        assert not (total_loss.isinf() or total_loss.isnan()), f'{total_loss=}'
        return ans

    def fit(self, X: np.ndarray, y: np.ndarray):
        self._module = nn.Linear(X.shape[1], 1, bias=True)
        self._module.train()
        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y).float()

        opt = torch.optim.LBFGS(self._module.parameters(), max_iter=self.max_iter,
                                tolerance_change=1e-6, line_search_fn='strong_wolfe')
        def closure():
            opt.zero_grad()
            ret = self._step((X,y))
            ret['loss'].backward()
            return ret['loss']

        with torch.autograd.set_detect_anomaly(True):
            for _ in range(self.max_iter):
                opt.step(closure)

    def decision_function(self, X):
        assert self._module
        self._module.eval()
        with torch.no_grad():
            scores = self._module(torch.from_numpy(X))
            return scores.detach().cpu().numpy()