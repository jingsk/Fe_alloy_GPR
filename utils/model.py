import gpytorch
import torch
from botorch.models.transforms.input import InputTransform
from torch import Tensor
from torch.nn import Module 
import numpy as np

def trainGP(model, mll, optimizer, num_epochs, print_interval =100):
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
        loss.backward()
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
        posterior = model.posterior(torch.tensor(X, dtype=torch.double))
    return posterior.mean.squeeze().numpy(), posterior.stddev.numpy()

def should_normalize(t, eps=1e-6):
    return torch.max(t) - torch.min(t) > eps

def normalize_to_0_1(t, eps=1e-6):
    return (t - torch.min(t)) / (torch.max(t) - torch.min(t) + eps)

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
        #check torch.normalize
        for idx in self.indices:
            if should_normalize(X[:,idx]):
                #print('normalizing')
                X[:,idx] = normalize_to_0_1(X[:,idx])
            #print(X)
            #print(torch.min(X[:,idx]))
            #print(torch.max(X[:,idx]))
        return X

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
        norm_factor = torch.sum(X[:,self.indices], axis=1)
        X[:, self.indices] = X[:, self.indices]/norm_factor[:,None]
        return X

def test_features_normalized(model, indices,eps=1e-5):
    X = model.train_inputs[0]
    X_mod = X.clone().detach()
    X_mod[:,indices] = 2*X_mod[:,indices]
    y_predicted1, y_train_stddev1 = evaluateGP(model, X_mod)
    y_predicted2, y_train_stddev2 = evaluateGP(model, X)
    return np.sum(y_predicted2-y_predicted1) < eps


