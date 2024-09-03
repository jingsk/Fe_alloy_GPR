import gpytorch
import torch
from botorch.models.transforms.input import InputTransform
from torch import Tensor
from torch.nn import Module 
import numpy as np

def trainGP(model, X_train, mll, optimizer, num_epochs):
    #num_epochs = 3000
    for epoch in range(num_epochs):
        # clear gradients
        optimizer.zero_grad()
        # forward pass through the model to obtain the output MultivariateNormal
        output = model(X_train)
        # Compute negative marginal log likelihood
        try:
            loss = - mll(output, model.train_targets)
        except:
            print("[{}] Couldn't train".format(epoch),flush=True)
            break
        # back prop gradients
        loss.backward()
        # print every X iterations
        if True and ((epoch + 1) % 500 == 0):
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
        posterior = model.posterior(X)
    return posterior.mean.squeeze().numpy(), posterior.stddev.numpy()


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
        norm_factor = torch.sum(X[:,self.indices], axis=1)
        return X[:,self.indices]/norm_factor[:,None]