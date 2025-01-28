import torch
import torch.nn as nn
from torch.utils.data import Dataset
import h5py
import numpy as np

class BaseModule(nn.Module):
    """Represents a `Base Module` that contains the basic functionality of an artificial neural network (ANN).

    All modules should inherit from :class:`.models.BaseModule` and override :meth:`vrnn.models.BaseModule.forward`.
    The Base Module itself inherits from :class:`torch.nn.Module`. See the `PyTorch` documentation for further information.
    """

    def __init__(self):
        """Constructor of the class. Initialize the Base Module.

        Should be called at the beginning of the constructor of a subclass.
        The class :class:`vrnn.models.BaseModule` should not be instantiated itself, but only its subclasses.
        """
        super().__init__()
        self.device = 'cpu'
        self.dtype = torch.float32

    def forward(*args):
        """Forward propagation of the ANN. Subclasses must override this method.

        :raises NotImplementedError: If the method is not overriden by a subclass.
        """
        raise NotImplementedError('subclasses must override forward()!')

    def training_step(self, dataloader, loss_fn, optimizer):
        """Single training step that performs the forward propagation of an entire batch,
        the training loss calculation and a subsequent optimization step.

        A training epoch must contain one call to this method.

        Example:
            >>> train_loss = module.training_step(train_loader, loss_fn, optimizer)

        :param dataloader: Dataloader with training data
        :type dataloader: :class:`torch.utils.data.Dataloader`
        :param loss_fn: Loss function for the model training
        :type loss_fn: method
        :param optimizer: Optimizer for model training
        :type optimizer: :class:`torch.optim.Optimizer`
        :return: Training loss
        :rtype: float
        """
        self.train()  # enable training mode
        cumulative_loss = 0
        samples = 0
        for x, y in dataloader:
            x, y = x.to(self.device, dtype=self.dtype), y.to(self.device, dtype=self.dtype)

            # Loss calculation
            optimizer.zero_grad()
            y_pred = self(x)
            loss = loss_fn(y_pred, y)
            cumulative_loss += loss.item() * x.size(0)
            samples += x.size(0)

            # Backpropagation
            loss.backward()
            optimizer.step()

        average_loss = cumulative_loss / samples
        return average_loss

    def loss_calculation(self, dataloader, loss_fns):
        """Perform the forward propagation of an entire batch from a given `dataloader`
        and the subsequent loss calculation for one or multiple loss functions in `loss_fns`.

        Example:
            >>> val_loss = module.loss_calculation(val_loader, loss_fn)

        :param dataloader: Dataloader with validation data
        :type dataloader: :class:`torch.utils.data.Dataloader`
        :param loss_fn: Loss function for model training
        :type loss_fn: method or list of methods
        :return: Validation loss
        :rtype: float
        """
        self.eval()  # disable training mode
        if not isinstance(loss_fns, list):
            loss_fns = [loss_fns]
        cumulative_loss = torch.zeros(len(loss_fns))
        samples = 0
        with torch.inference_mode():  # disable gradient calculation
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                y_pred = self(x)
                samples += x.size(0)
                for i, loss_fn in enumerate(loss_fns):
                    loss = loss_fn(y_pred, y)
                    cumulative_loss[i] += loss.item() * x.size(0)
        average_loss = cumulative_loss / samples
        if torch.numel(average_loss) == 1:
            average_loss = average_loss[0]
        return average_loss

    def parameter_count(self):
        """Get the number of learnable parameters, that are contained in the model.

        :return: Number of learnable parameters
        :rtype: int
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def reduce(cls, loss, reduction='mean'):
        """Perform a reduction step over all datasets to transform a loss function to a cost function.

        A loss function is evaluated element-wise for a dataset.
        However, a cost function should return a single value for the dataset.
        Typically, `mean` reduction is used.

        :param loss: Tensor that contains the element-wise loss for a dataset
        :type loss: :class:`torch.Tensor`
        :param reduction: ('mean'|'sum'), defaults to 'mean'
        :type reduction: str, optional
        :return: Reduced loss
        :rtype: float
        """
        if reduction == 'mean':
            return torch.mean(loss)
        elif reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss

    @classmethod
    def unsqueeze(cls, output, target):
        """Ensure that the tensors :code:`output` and :code:`target` have a shape of the form :code:`(N, features)`.

        When a loss function is called with a single data point, the tensor shape is :code:`(features)` and hence does not fit.
        This method expands the dimensions if needed.

        :param output: Model output
        :type output: :class:`torch.Tensor`
        :param target: Target data
        :type target: :class:`torch.Tensor`
        :return: Tuple (output, target)
        :rtype: tuple
        """
        if output.dim() == 1:
            output = torch.unsqueeze(output, 0)
        if target.dim() == 1:
            target = torch.unsqueeze(target, 0)
        return output, target

    def to(self, device, dtype=None, *args, **kwargs):
        """Transfers a model to another device, e.g. to a GPU.

        This method overrides the PyTorch built-in method :code:`model.to(...)`.

        Example:
            >>> # Transfer the model to a GPU
            >>> module.to('cuda:0')

        :param device: Identifier of the device, e.g. :code:`'cpu'`, :code:`'cuda:0'`, :code:`'cuda:1'`, ...
        :type device: str
        :return: The model itself
        :rtype: :class:`vrnn.models.BaseModule`
        """
        self.device = device

        if dtype is None:
            return super().to(device=device, *args, **kwargs)
        else:
            self.dtype = dtype
            return super().to(device=device, dtype=dtype, *args, **kwargs)

    @property
    def gpu(self):
        """Property, that indicates, whether the model is on a GPU, i.e., not on the CPU.

        :return: True, iff the module is not on the CPU
        :rtype: bool
        """
        return self.device != 'cpu'

class VanillaModule(BaseModule):
    def __init__(self, ann_module: torch.nn.Module, dim: int = 3):
        super().__init__()
        self.ann_module = ann_module
        self.dim = dim

    def forward(self, x):
        return self.ann_module(x)
        

class Dataset2DThermal(Dataset):
    def __init__(self, file_name, R_range, group, feature_idx=None):
        self.file_name = file_name
        self.R_range = R_range
        self.group = group

        if feature_idx is None:
            feature_idx = slice(None)

        self.feature_idx = feature_idx
        self.features, self.kappa = self.load_data()
    
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.kappa[idx]

    def load_data(self):
        features_list = []
        kappa_list = []

        with h5py.File(self.file_name, "r") as F:
            feature_vectors = torch.tensor(F[f"{self.group}/feature_vector"][...], dtype=torch.float32)

            # Truncate feature vector
            feature_vectors = feature_vectors[..., self.feature_idx]

            num_samples = feature_vectors.shape[0]

            for R in self.R_range:
                R_column = torch.ones((num_samples, 1)) * R
                onebyR_column = torch.ones((num_samples, 1)) / R
                features_with_R = torch.hstack((feature_vectors, onebyR_column, R_column))
                features_list.append(features_with_R)

                if R < 1:  # For fractional R, the corresponding kappa is stored as contrast_{R_value}_inv
                    R_key = int(round(1 / R))
                    kappa_grp = torch.tensor(F[f"{self.group}/effective_heat_conductivity/contrast_{R_key}_inv"][...], dtype=torch.float32)
                elif R > 1:
                    R_key = int(round(R))
                    kappa_grp = torch.tensor(F[f"{self.group}/effective_heat_conductivity/contrast_{R_key}"][...], dtype=torch.float32)
                else:
                    kappa_grp = torch.ones((num_samples, 3))
                    kappa_grp[..., 2] = 0.
                kappa_list.append(kappa_grp)

        # Concatenate all features and kappa arrays vertically
        all_features = torch.vstack(features_list)
        all_kappa = torch.vstack(kappa_list)
        all_kappa[:, 2] = all_kappa[:, 2] / np.sqrt(2.0)

        # Convert to PyTorch tensors
        features_tensor = all_features
        kappa_tensor = all_kappa
        
        #Check all values are finite
        if not torch.all(torch.isfinite(features_tensor)):
            raise ValueError("Invalid features_tensor: contains non-finite values in group {}".format(self.group))
        if not torch.all(torch.isfinite(kappa_tensor)):
            raise ValueError("Invalid kappa_tensor: contains non-finite values in group {}".format(self.group))

        return features_tensor, kappa_tensor
    
# Functions for converting between symmetric matrix representations

def get_sym_indices(dim):
    diag_idx = (torch.arange(dim), torch.arange(dim))    
    row, col = torch.tril_indices(dim, dim, -1)
    dof_idx = (torch.cat([diag_idx[0], row]), torch.cat([diag_idx[1], col]))
    return dof_idx

def pack_sym(symmetric_matrix, dim, dof_idx=None):
    if dof_idx is None:
        dof_idx = get_sym_indices(dim)
    dof_idx = tuple(idx.to(symmetric_matrix.device) for idx in dof_idx)
    return symmetric_matrix[(..., *dof_idx) if symmetric_matrix.dim() == 3 else dof_idx]

def unpack_sym(packed_values, dim, dof_idx=None):
    if dof_idx is None:
        dof_idx = get_sym_indices(dim)
    dof_idx = tuple(idx.to(packed_values.device) for idx in dof_idx)
    matrix = torch.zeros((*packed_values.shape[:-1], dim, dim), dtype=packed_values.dtype, device=packed_values.device)
    if packed_values.dim() == 2:
        matrix[:, dof_idx[0], dof_idx[1]] = packed_values
        return matrix + matrix.transpose(1, 2) - torch.diag_embed(torch.diagonal(matrix, dim1=1, dim2=2))
    matrix[dof_idx] = packed_values
    return matrix + matrix.T - torch.diag(torch.diag(matrix))