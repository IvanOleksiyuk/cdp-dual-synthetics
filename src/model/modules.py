"""Collection of pytorch modules that make up the common networks."""

import math

import torch as T
import torch.nn as nn
from torch.autograd import Function


class IterativeNormLayer(nn.Module):
    """A basic normalisation layer so it can be part of the model.

    Tracks the runnning mean and variances of an input vector over the batch dimension.
    Must always be passed batched data!
    Any additional dimension to calculate the stats must be provided as dims.

    For example: Providing an image with inpt_dim = channels, width, height
    Will result in an operation with independant stats per pixel, per channel
    If instead you want the mean and shift only for each channel you will have to give
    dims = (1, 2) or (-2, -1)
    This will tell the layer to average out the width and height dimensions

    Note! If a mask is provided in the forward pass, then this must be
    the dimension to apply over the masked inputs! For example: Graph
    nodes are usually batch x n_nodes x features so to average out the n_nodes
    one would typically give dims as (0,). But nodes
    are always passed with the mask which flattens it to batch x features.
    Batch dimension is done automatically, so we dont pass any dims!!!
    """

    def __init__(
        self,
        inpt_dim: T.Tensor | tuple | int,
        means: T.Tensor | None = None,
        vars: T.Tensor | None = None,
        n: int = 0,
        max_n: int = 1_00_000,
        dims: tuple | int = (),
        track_grad_forward: bool = True,
        track_grad_reverse: bool = False,
        ema_sync: float = 0.0,
    ) -> None:
        """Init method for Normalisatiion module.

        Parameters
        ----------
        inpt_dim : int
            Shape of the input tensor (non batched), required for reloading.
        means : float, optional
            Calculated means for the mapping. Defaults to None.
        vars : float, optional
            Calculated variances for the mapping. Defaults to None.
        n : int, optional
            Number of samples already used to make the mapping. Defaults to 0.
        max_n : int
            Maximum number of iterations before the means and vars are frozen.
        dims : int
            The dimension(s) over which to calculate the stats.
            Dimensions must be expressed for non-batched data!
            Will always calculate over the batch dimension!
        track_grad_forward : bool
            If the gradients should be tracked for this operation.
        track_grad_reverse : bool
            If the gradients should be tracked for this operation.
        ema_sync : bool
            If we should use an exponential moving average.
        """
        super().__init__()

        # Fail if only one of means or vars is provided
        if (means is None) ^ (vars is None):  # XOR
            raise ValueError(
                """Only one of 'means' and 'vars' is defined. Either both or
                neither must be defined"""
            )
        if means is not None and means.shape != vars.shape:
            raise ValueError("The shapes of 'means' and 'vars' do not match")

        # Modify the dims to be compatible with the layer
        dims, inpt_dim, n = self._modify_dims(dims, inpt_dim, n)

        # Class attributes
        self.stat_dim = self._calculate_stat_dim(inpt_dim, dims, means)
        self.inpt_dim = inpt_dim
        self.dims = dims
        self.max_n = max_n
        self.ema_sync = ema_sync
        self.do_ema = ema_sync > 0
        self.track_grad_forward = track_grad_forward
        self.track_grad_reverse = track_grad_reverse

        # Buffers are needed for saving/loading the layer
        self.register_buffer(
            "means",
            T.zeros(self.stat_dim, dtype=T.float32)
            if means is None
            else T.as_tensor(means, dtype=T.float32),
        )
        self.register_buffer(
            "vars",
            T.ones(self.stat_dim, dtype=T.float32)
            if vars is None
            else T.as_tensor(vars, dtype=T.float32),
        )
        self.register_buffer("n", n)

        # For the welford algorithm it is useful to have another variable m2
        self.register_buffer(
            "m2",
            T.ones(self.stat_dim, dtype=T.float64)
            if vars is None
            else T.as_tensor(vars, dtype=T.float64),
        )

        # If the means are set here then the model is "frozen" and never updated
        self.register_buffer(
            "frozen",
            T.as_tensor(
                (means is not None and vars is not None) or self.n > self.max_n
            ),
        )

    def _modify_dims(
        self, dims: tuple | int, inpt_dim: T.Tensor | tuple | int, n: int
    ) -> tuple:
        """Modify the dims argument to be compatible with the layer."""

        # Input dim must be a list
        if isinstance(inpt_dim, int):
            inpt_dim = [inpt_dim]
        else:
            inpt_dim = list(inpt_dim)

        # N must be a tensor
        if isinstance(n, int):
            n = T.tensor(n)

        # Dims must be a list
        if isinstance(dims, int):
            dims = [dims]
        else:
            dims = list(dims)

        # Check the dims are within the input range
        if any([abs(d) > len(inpt_dim) for d in dims]):
            raise ValueError("Dims argument lists dimensions outside input shape")

        # Convert negative dims to positive
        for d in range(len(dims)):
            if dims[d] < 0:  # make positive
                dims[d] = len(inpt_dim) + dims[d]
            dims[d] += 1  # Add one because we are inserting a batch dimension

        return dims, inpt_dim, n

    def _calculate_stat_dim(
        self, inpt_dim: list, dims: list, means: T.Tensor | None
    ) -> tuple:
        """Calculate the shape of the statistics."""

        # If the means are defined just use their shape
        if means is not None:
            return means.shape

        # Calculate the input and output shapes
        stat_dim = [1] + list(inpt_dim)  # Add batch dimension
        for d in range(len(stat_dim)):
            if d in dims:
                stat_dim[d] = 1
        return stat_dim

    def __repr__(self):
        return f"IterativeNormLayer({list(self.means.shape)})"

    def __str__(self) -> str:
        return f"IterativeNormLayer(m={self.means.squeeze()}, v={self.vars.squeeze()})"

    def _mask(self, inpt: T.Tensor, mask: T.BoolTensor | None = None) -> T.Tensor:
        if mask is None:
            return inpt
        return inpt[mask]

    def _unmask(
        self, inpt: T.Tensor, output: T.Tensor, mask: T.BoolTensor | None = None
    ) -> T.Tensor:
        if mask is None:
            return output
        masked_out = inpt.clone()  # prevents inplace operation, bad for autograd
        masked_out[mask] = output.type(masked_out.dtype)
        return masked_out

    def fit(
        self, inpt: T.Tensor, mask: T.BoolTensor | None = None, freeze: bool = True
    ) -> None:
        """Set the stats given a population of data."""
        cur_mean_shape = self.means.shape
        inpt = self._mask(inpt, mask)
        self.vars, self.means = T.var_mean(inpt, dim=(0, *self.dims), keepdim=True)
        self.n = T.tensor(len(inpt), device=self.means.device)
        self.m2 = (self.vars * self.n).type(self.m2.dtype)
        if freeze:
            self.frozen.fill_(True)
        self._check_shape_change(cur_mean_shape)

    def _check_shape_change(self, cur_mean_shape: tuple) -> None:
        if not cur_mean_shape == self.means.shape:
            print(f"WARNING! The stats in {self} have changed shape!")
            print("This could be due to incorrect initialisation or masking")
            print(f"Old shape: {cur_mean_shape}")
            print(f"New shape: {self.means.shape}")

    def forward(self, inpt: T.Tensor, mask: T.BoolTensor | None = None) -> T.Tensor:
        """Apply standardisation to a batch of inputs.

        Uses the inputs to update the running stats if in training mode.
        """
        # Save and check the gradient tracking options
        grad_setting = T.is_grad_enabled()
        T.set_grad_enabled(self.track_grad_forward and grad_setting)

        # Mask the inputs and update the stats
        sel_inpt = self._mask(inpt, mask)

        # Only update if in training mode
        if self.training:
            self.update(sel_inpt)

        # Apply the mapping
        normed_inpt = (sel_inpt - self.means) / (self.vars.sqrt() + 1e-8)

        # Undo the masking
        normed_inpt = self._unmask(inpt, normed_inpt, mask)

        # Revert the gradient setting
        T.set_grad_enabled(grad_setting)

        return normed_inpt

    def reverse(self, inpt: T.Tensor, mask: T.BoolTensor | None = None) -> T.Tensor:
        """Unnormalises the inputs given the recorded stats."""

        # Save and check the gradient tracking options
        grad_setting = T.is_grad_enabled()
        T.set_grad_enabled(self.track_grad_reverse and grad_setting)

        # Mask, revert the inputs, unmask
        sel_inpt = self._mask(inpt, mask)
        unnormed_inpt = sel_inpt * self.vars.sqrt() + self.means
        unnormed_inpt = self._unmask(inpt, unnormed_inpt, mask)

        # Revert the gradient setting
        T.set_grad_enabled(grad_setting)

        return unnormed_inpt

    @T.no_grad()
    def update(self, inpt: T.Tensor) -> None:
        """Update the running stats using a batch of data."""

        # Check the current shapes of the means
        cur_mean_shape = self.means.shape

        # Freeze the model if we already exceed the requested stats
        T.fill_(self.frozen, self.n >= self.max_n)
        if self.frozen:
            return

        # For first iteration, just run the fit on the batch
        if self.n == 0:
            self.fit(inpt, freeze=False)
            return

        # Otherwise update the statistics
        if self.do_ema:
            self._apply_ema_update(inpt)
        else:
            self._apply_welford_update(inpt)
        
        # Check if the shapes changed
        self._check_shape_change(cur_mean_shape)

    @T.no_grad()
    def _apply_ema_update(self, inpt: T.Tensor) -> None:
        """Use an exponential moving average to update the means and vars."""
        self.n += len(inpt)
        nm = inpt.mean(dim=(0, *self.dims), keepdim=True)
        self.means = self.ema_sync * self.means + (1 - self.ema_sync) * nm
        nv = (inpt - self.means).square().mean((0, *self.dims), keepdim=True)
        self.vars = self.ema_sync * self.vars + (1 - self.ema_sync) * nv

    @T.no_grad()
    def _apply_welford_update(self, inpt: T.Tensor) -> None:
        """Use the welford algorithm to update the means and vars."""
        m = len(inpt)
        d = inpt - self.means
        self.n += m
        self.means += (d / self.n).mean(dim=(0, *self.dims), keepdim=True) * m
        d2 = inpt - self.means
        self.m2 += (d * d2).mean(dim=(0, *self.dims), keepdim=True) * m
        self.vars = (self.m2 / self.n).to(self.vars.dtype)
