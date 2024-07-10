import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from .basic_trainer import BasicTrainer
import numpy as np
import pandas as pd
import math


class LinearModel:
    def __init__(self, *,
         label_loss_type : str = 'hinge_squared_loss',
         reg_norm_lambda : float = 10.0, # equiv to C = .1 in sklearn
         regularizer_vector : np.ndarray | None = None,
         reg_vector_lambda : float = 1.0,
         class_weight : str = 'balanced',
         verbose : bool = False,
         max_iter : int = 100
        ):

        self.regularizer_vector = torch.from_numpy(regularizer_vector).float() if regularizer_vector is not None else None
        assert label_loss_type in ['ce_loss', 'hinge_loss', 'hinge_squared_loss']
        self.label_loss_type = label_loss_type
        self._module : nn.Linear | None = None
        self.max_iter = max_iter
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
#       normalized_weight = F.normalize(self.weight, dim=-1)
#       loss_norm = self.reg_norm_lambda *  ( torch.cosh( (self.weight @ self.weight).log() ) - 1. )
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

    def training_step(self, batch, batch_idx):
        losses = self._step(batch)
        return losses

    def validation_step(self, batch, batch_idx):
        losses = self._step(batch)
        return losses

    def configure_optimizers(self):
        return opt.LBFGS(self._module.parameters(), max_iter=self.max_iter, tolerance_change=1e-6, line_search_fn='strong_wolfe')

    def fit(self, X, y):
        self._module = nn.Linear(X.shape[1], 1, bias=True)
        trainer_ = BasicTrainer(mod=self, max_epochs=1, verbose=self.verbose)

        assert X.shape[0] > 0
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        dl = DataLoader(ds, batch_size=X.shape[0], shuffle=True)
        losses_ = trainer_.fit(dl)
        if self.verbose:
            df = pd.DataFrame.from_records(losses_)
            agg_df= df.groupby('k').mean()
            print(agg_df)
        return losses_

    def decision_function(self, X):
        assert self._module
        with torch.no_grad():
            scores = self._module(torch.from_numpy(X))
            return scores.detach().cpu().numpy()