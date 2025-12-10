import threading
from abc import ABC
from typing import Optional, Union

import gpytorch
import numpy as np
import torch

from ..pytorch_feature_selector import PyTorchFeatureSelector
from ..pytorch_utils import to_numpy, to_tensor
from .gpytorch_gp import BatchIndependentApproximateSpatioTemporalGPModel


class DataProcessingStrategy(ABC):
    """Abstract base class for the data processing strategies.

    Depending on the experiment/operating mode of the controller, the data processing
    strategy might be different. To enable as much flexibility as possible, the data
    processing strategy is impelented using a strategy pattern, allowing the user to
    easily define their own strategies and swap them out when necessary
    """

    def process(
        self,
        gp_model: gpytorch.models.GP,
        x_input: Union[np.ndarray, torch.Tensor],
        y_target: Union[np.ndarray, torch.Tensor],
        gp_feature_selector: PyTorchFeatureSelector,
        timestamp: Optional[float],
    ) -> Optional[gpytorch.models.GP]:
        """Function which is processed in the 'record_datapoint' method of the 'GPyTorchResidualModel'.

        Args:
            - gp_model: Instance of the residual GP model class so we can access
              relevant attributes
            - x_input: data which should be saved. Should have dimension (state_dimension,) or equivalent
            - y_target: the residual which was measured at x_input. Should have dimension
              (residual_dimension,) or equivalent
            - gp_feature_selector: 'FeatureSelector' instance to select the relevant GP input features
              from the 'x_input'
            - timestamp: Optional timestamp of the data point
        """
        raise NotImplementedError


class VoidDataStrategy(DataProcessingStrategy):
    def process(
        self,
        gp_model: gpytorch.models.ExactGP,
        x_input: Union[np.ndarray, torch.Tensor],
        y_target: Union[np.ndarray, torch.Tensor],
        gp_feature_selector: PyTorchFeatureSelector,
        timestamp: Optional[float],
    ) -> Optional[gpytorch.models.ExactGP]:
        pass


class RecordDataStrategy(DataProcessingStrategy):
    """Implements a processing strategy which saves the data continuously to a file.

    The strategy keeps a buffer of recent datapoints and asynchronously saves the buffer to
    a file.
    """

    def __init__(self, x_data_path: str, y_data_path: str, buffer_size: int = 50):
        """Construct the data recorder.

        Args:
            - x_data_path: file path where x data should be saved.
            - y_data_path: file path where residual data should be saved
        """
        self.x_data_path = x_data_path
        self.y_data_path = y_data_path
        self.buffer_size = buffer_size
        self._gp_training_data = {"x_training_data": [], "y_training_data": []}

    def process(
        self,
        gp_model: gpytorch.models.ExactGP,
        x_input: Union[np.ndarray, torch.Tensor],
        y_target: Union[np.ndarray, torch.Tensor],
        gp_feature_selector: PyTorchFeatureSelector,
        timestamp: Optional[float],
    ) -> Optional[gpytorch.models.ExactGP]:

        # Convert to numpy array
        if torch.is_tensor(x_input):
            x_input = to_numpy(x_input, x_input.device)
        if torch.is_tensor(y_target):
            y_target = to_numpy(y_target, y_target.device)

        self._gp_training_data["x_training_data"].append(x_input)
        self._gp_training_data["y_training_data"].append(y_target)

        if len(self._gp_training_data["x_training_data"]) == self.buffer_size:
            # Do we need a local copy?
            save_data_x = np.array(self._gp_training_data["x_training_data"])
            save_data_y = np.array(self._gp_training_data["y_training_data"])

            self._gp_training_data["x_training_data"].clear()
            self._gp_training_data["y_training_data"].clear()

            threading.Thread(
                target=lambda: (
                    RecordDataStrategy._save_to_file(save_data_x, self.x_data_path),
                    RecordDataStrategy._save_to_file(save_data_y, self.y_data_path),
                    print("saved gp training data"),
                )
            ).start()

    @staticmethod
    def _save_to_file(data: np.ndarray, filename: str) -> None:
        """Appends data to a file"""
        with open(filename, "ab") as f:
            # f.write(b"\n")
            np.savetxt(
                f,
                data,
                delimiter=",",
            )


class OnlineLearningStrategy(DataProcessingStrategy):
    """Implements an online learning strategy.

    The received data is incorporated in the GP and used for further predictions.
    This data processing strategy depends on the [online_gp] optional dependencies (see pyproject.toml).
    """

    def __init__(
        self,
        max_num_points: int = 200,
        data_selection: str = "balanced",
        device: str = "cpu",
    ) -> None:
        self.max_num_points = max_num_points
        if data_selection not in ("newest", "random", "balanced"):
            raise ValueError("Data selection must be 'newest', 'random', or 'balanced'.")
        self.data_selection = data_selection
        self.use_newest = (data_selection == "newest")
        self.device = device

    def process(
        self,
        gp_model: gpytorch.models.ExactGP,
        x_input: Union[np.ndarray, torch.Tensor],
        y_target: Union[np.ndarray, torch.Tensor],
        gp_feature_selector: PyTorchFeatureSelector,
        timestamp: Optional[float] = None,
    ) -> Optional[gpytorch.models.ExactGP]:

        # Convert to tensor
        if not torch.is_tensor(x_input):
            x_input, _ = to_tensor(arr=x_input, device=self.device)

        if not torch.is_tensor(y_target):
            y_target, _ = to_tensor(arr=y_target, device=self.device)

        # Extend to 2D for further computation
        x_input = torch.atleast_2d(x_input)
        y_target = torch.atleast_2d(y_target)

        if (
            gp_model.prediction_strategy is None
            or gp_model.train_inputs is None
            or gp_model.train_targets is None
        ):
            if gp_model.train_inputs is not None:
                raise RuntimeError(
                    "train_inputs in GP is not None. Something went wrong."
                )

            # Set the training data and return (in-place modification)
            gp_model.set_train_data(
                gp_feature_selector(x_input, timestamp=timestamp),
                y_target,
                strict=False,
            )
            return

        X = gp_model.train_inputs[0]
        x_new = gp_feature_selector(x_input, timestamp=timestamp)

        balanced_drop_idx = None
        if self.data_selection == "balanced":
            accept, balanced_drop_idx = self._balanced_gate_and_drop(X, x_new)
            if not accept:
                return None

        # Check if GP is already full
        if X.shape[-2] >= self.max_num_points:
            with torch.no_grad():
                if self.data_selection == "balanced" and balanced_drop_idx is not None:
                    drop_idx = int(balanced_drop_idx)
                elif self.use_newest:
                    drop_idx = 0
                else:
                    drop_idx = torch.randint(0, self.max_num_points, torch.Size(), requires_grad=False).item()

                selector = torch.ones(self.max_num_points, requires_grad=False)
                selector[drop_idx] = 0

                try:
                    fantasy_model = gp_model.get_fantasy_model(
                        x_new, y_target, data_selector=selector
                    )
                except TypeError as err:
                    if "data_selector" in str(err):
                        raise ImportError(
                            "OnlineLearningStrategy requires the [gpytorch-exo] optional dependencies (see pyproject.toml)."
                        )
                    raise err

                return fantasy_model

        with torch.no_grad():
            # Add observation and return updated model
            fantasy_model = gp_model.get_fantasy_model(
                x_new, y_target
            )

            return fantasy_model

    def _balanced_gate_and_drop(self, X: torch.Tensor, x_new: torch.Tensor):
        if X.shape[-2] < 2:
            return True, None

        D_xx = torch.cdist(X, X, p=2)               # (N, N)
        N = D_xx.size(0)
        D_xx.fill_diagonal_(float("inf"))
        d_min_existing, flat_idx = torch.min(D_xx.view(-1), dim=0)
        i_min = (flat_idx // N).item()
        j_min = (flat_idx % N).item()

        d_min_new = torch.cdist(x_new, X, p=2).min()
        accept = bool(d_min_new > d_min_existing)
        drop_idx = j_min if accept else None
        return accept, drop_idx


class KalmanLearningStrategy(DataProcessingStrategy):
    """Implements an online learning strategy based on Kalman filter updates.

    This strategy requires the gp_model of the residual_gp_instance to have an update method that has
    the Kalman filter equations implemented (such as in the 'BatchIndependentApproximateSpatioTemporalGPModel' class).
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = device

    def process(
        self,
        gp_model: BatchIndependentApproximateSpatioTemporalGPModel,
        x_input: Union[np.ndarray, torch.Tensor],
        y_target: Union[np.ndarray, torch.Tensor],
        gp_feature_selector: PyTorchFeatureSelector,
        timestamp: Optional[float],
    ) -> None:

        # Convert to tensor
        if not torch.is_tensor(x_input):
            x_input, _ = to_tensor(arr=x_input, device=self.device)
        if not torch.is_tensor(y_target):
            y_target, _ = to_tensor(arr=y_target, device=self.device)

        # Extend to 2D for further computation
        x_input = torch.atleast_2d(x_input)
        y_target = torch.atleast_2d(y_target)

        gp_model.update(gp_feature_selector(x_input, timestamp=timestamp), y_target)

        return
