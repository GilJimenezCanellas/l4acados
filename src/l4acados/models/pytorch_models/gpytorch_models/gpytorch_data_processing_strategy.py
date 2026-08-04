import math
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
        min_dist: float = 0.15,
        feature_scales: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> None:
        """
        Args:
            max_num_points: dictionary capacity. Once full, an accepted point
                replaces one member of the closest existing pair.
            data_selection: which point to evict when full -- "newest", "random",
                or "balanced" (evict from the closest pair).
            min_dist: admission threshold for "balanced", as a per-feature RMS
                distance in units of each feature's standard deviation. A candidate
                is admitted only if it is at least this far from every point already
                in the dictionary.
            feature_scales: per-feature scales used to normalise that distance. If
                omitted, a running standard deviation over every candidate seen is
                used, which makes min_dist independent of the units and ranges of
                the individual features.
        """
        self.max_num_points = max_num_points
        if data_selection not in ("newest", "random", "balanced"):
            raise ValueError("Data selection must be 'newest', 'random', or 'balanced'.")
        self.data_selection = data_selection
        self.use_newest = (data_selection == "newest")
        self.device = device
        if not min_dist > 0.0:
            raise ValueError("min_dist must be positive.")
        self.min_dist = float(min_dist)
        self._fixed_scales = (
            None if feature_scales is None else to_tensor(arr=feature_scales, device=device)[0]
        )
        # Running per-feature moments over every candidate seen (Welford), so the
        # threshold is scale free without the caller having to supply scales.
        self._n_seen = 0
        self._mean = None
        self._m2 = None

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
            x_first = gp_feature_selector(x_input, timestamp=timestamp)
            self._observe(x_first)
            gp_model.set_train_data(
                x_first,
                y_target,
                strict=False,
            )
            return

        X = gp_model.train_inputs[0]
        x_new = gp_feature_selector(x_input, timestamp=timestamp)
        self._observe(x_new)

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

    def _observe(self, x_new: torch.Tensor) -> None:
        """Welford update of the per-feature moments used to normalise distances."""
        x = x_new.reshape(-1, x_new.shape[-1]).detach()
        for row in x:
            self._n_seen += 1
            if self._mean is None:
                self._mean = torch.zeros_like(row)
                self._m2 = torch.zeros_like(row)
            delta = row - self._mean
            self._mean = self._mean + delta / self._n_seen
            self._m2 = self._m2 + delta * (row - self._mean)

    def _feature_scale(self, X: torch.Tensor) -> torch.Tensor:
        """Per-feature scale for the admission metric; never zero."""
        if self._fixed_scales is not None:
            scale = self._fixed_scales.to(device=X.device, dtype=X.dtype)
        elif self._n_seen > 1 and self._m2 is not None:
            scale = torch.sqrt(self._m2 / (self._n_seen - 1)).to(device=X.device, dtype=X.dtype)
        else:
            scale = torch.ones(X.shape[-1], device=X.device, dtype=X.dtype)
        return torch.clamp(scale, min=1e-9)

    def _balanced_gate_and_drop(self, X: torch.Tensor, x_new: torch.Tensor):
        """Admit a candidate only if it is at least ``min_dist`` from the dictionary.

        The threshold is absolute (in normalised feature-std units) on purpose. It
        used to be the dictionary's own smallest pairwise distance, which cannot
        decrease when a point is admitted and is only ever recomputed after an
        eviction -- and evictions only happen once the dictionary is full. Below
        capacity the bar was therefore frozen at whatever gap the first two admitted
        samples happened to have, so the final dictionary size was decided by the
        sampling period and by which two points arrived first: replaying the rule on
        one fixed trajectory gave 137 points at 50 Hz but 30 at 16.7 Hz, and 30 vs 94
        for the same data with a different starting offset.
        """
        scale = self._feature_scale(X)
        Xs, xs = X / scale, x_new / scale
        rms = math.sqrt(X.shape[-1])  # so min_dist is per-feature, not per-vector

        if bool((torch.cdist(xs, Xs, p=2).min() / rms) <= self.min_dist):
            return False, None

        if X.shape[-2] < 2:
            return True, None

        D_xx = torch.cdist(Xs, Xs, p=2)             # (N, N)
        N = D_xx.size(0)
        D_xx.fill_diagonal_(float("inf"))
        _, flat_idx = torch.min(D_xx.view(-1), dim=0)
        return True, (flat_idx % N).item()


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
