import gpytorch
import torch

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