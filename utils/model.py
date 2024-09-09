import gpytorch
import torch
from botorch.models.transforms.input import InputTransform
from torch import Tensor
from torch.nn import Module 
import numpy as np
from torch.optim import Adam
from gpytorch.mlls.sum_marginal_log_likelihood import ExactMarginalLogLikelihood

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    #"device": torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"),
}

def trainGP(model, lr=3e-2):
    #optimizer_kwargs = {'lr': 1e-2, 'weight_decay': 1e-3}
    optimizer_kwargs = {'lr': lr, 'weight_decay': 1e-3}
    #print(f"Training {model}")
    #surrogate_model.model.train()
    _train_GPmodel(
        model,
        mll = ExactMarginalLogLikelihood(model.likelihood, model), 
        optimizer = Adam([{'params': model.parameters()}], **optimizer_kwargs),
        num_epochs=200,
        print_interval = 50 
    )
    print("-------------------")


def _train_GPmodel(model, mll, optimizer, num_epochs, print_interval =100):
    #num_epochs = 3000
    for epoch in range(num_epochs):
        # clear gradients
        optimizer.zero_grad()
        # forward pass through the model to obtain the output MultivariateNormal
        output = model(model.train_inputs[0])
        # Compute negative marginal log likelihood
        try:
            loss = - mll(output, model.train_targets)
        except:
            print("[{}] Couldn't train".format(epoch),flush=True)
            break
        # back prop gradients
        loss.backward(retain_graph=True)
        # print every X iterations
        if True and ((epoch + 1) % print_interval == 0):
            print(
                f"Epoch {epoch+1:>3}/{num_epochs} - Loss: {loss.item():>4.3f} "
                # f"lengthscale: {model.covar_module.base_kernel.lengthscale.item():>4.3f} " 
                f"noise: {model.likelihood.noise.item():>4.3f}"
            )
        optimizer.step()

def evaluateGP(model, X):
    model.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # compute posterior
        posterior = model.posterior(torch.tensor(X, **tkwargs))
    return posterior.mean.squeeze().numpy(), posterior.stddev.squeeze().numpy()

def should_normalize(t, eps=1e-6):
    return torch.max(t) - torch.min(t) > eps

def normalize_to_0_1(t, eps=1e-6):
    return (t - torch.min(t)) / (torch.max(t) - torch.min(t))

class NormalizeFeatures(InputTransform,Module):
    def __init__(
        self,
        #d,
        indices,
        transform_on_train: bool = True,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = True,
    ):
        super().__init__()
        self.indices = np.array(indices)
        self.transform_on_train = transform_on_train
        self.transform_on_eval = transform_on_eval
        self.transform_on_fantasize = transform_on_fantasize
    def transform(self, X: Tensor) -> Tensor:
        r"""Transform the inputs to a model.

        Args:
            X: A `batch_shape x n x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of transformed inputs.
        """
        X_transformed = X.detach().clone().to(tkwargs["device"])
        for idx in self.indices:
            if should_normalize(X_transformed[...,idx]):
                X_transformed[...,idx] = normalize_to_0_1(X_transformed[...,idx])
        return X_transformed

class NormalizeElementFractions(InputTransform,Module):
    def __init__(
        self,
        #d,
        indices,
        transform_on_train: bool = True,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = True,
    ):
        super().__init__()
        self.indices = np.array(indices)
        self.transform_on_train = transform_on_train
        self.transform_on_eval = transform_on_eval
        self.transform_on_fantasize = transform_on_fantasize
    def transform(self, X: Tensor) -> Tensor:
        r"""Transform the inputs to a model.

        Args:
            X: A `batch_shape x n x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of transformed inputs.
        """
        #check torch.normalize
        #print(X.shape)
        X_transformed = X.detach().clone().to(tkwargs["device"])
        if X_transformed.dim() ==2:
            X_transformed[:,self.indices] -= torch.tensor(np.min(X_transformed.numpy(),axis=1).repeat(X_transformed.shape[1]).reshape(-1,X_transformed.shape[1]), **tkwargs)[:,self.indices]
            norm_factor = torch.sum(X_transformed[:,self.indices], axis=1)
            X_transformed[:, self.indices] = X_transformed[:, self.indices]/norm_factor[:,None]
        # if X_transformed.dim() ==3:
        #     norm_factor = torch.sum(X_transformed[:,:,self.indices], axis=-1)
        #     X_transformed[:, :, self.indices] = X_transformed[:, :, self.indices]/norm_factor[:,:,None]
        
        return X_transformed

def test_features_normalized(model, indices,eps=1e-5):
    X = model.train_inputs[0]
    X_mod = X.clone().detach()
    X_mod[:,indices] = 2*X_mod[:,indices]
    y_predicted1, y_train_stddev1 = evaluateGP(model, X_mod)
    y_predicted2, y_train_stddev2 = evaluateGP(model, X)
    return np.sum(y_predicted2-y_predicted1) < eps


