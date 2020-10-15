import gzip
import os
from abc import abstractmethod, ABC
from itertools import islice
from tempfile import TemporaryDirectory
from typing import Any, Dict, Optional, Tuple, Iterator, TypeVar, Generic

import torch
from torch import nn
from typing_extensions import final  # For backwards compatibility with Py < 3.8

from dpu_utils.utils import RichPath

InputData = TypeVar('InputData')
TensorizedData = TypeVar('TensorizedData')

__all__ = ['BaseComponent']

class BaseComponent(ABC, nn.Module, Generic[InputData, TensorizedData]):
    """
    Implements the base class for neural network components in pyTorch. Each component has
    a few basic functionalities that all components should implement. All components are also
    PyTorch `nn.Module`s and hence maintain all of their methods.

     Specifically,

    * Metadata: All information that are needed to build the component that depend on the data.
        For example, a vocabulary of common words, a list of classes, etc.
    * Hyperparameters: All parameters that are not learned but reflect design decisions about
        the component.
    * Tensor Conversion: The way that any input data is converted into tensors that can then be
        input into the model. For example, a sentence is converted into a (padded) list of integer
        word ids.
    * Minibatching: How multiple input data are aggregated into a minibatch. Commonly, this
        is implemented as stacking of per-data tensors, but this is not always true.
    * Neural model: Only a tiny part of a component usually. Accepts a minibatch of
        tensor data (if any) along with input data from other components and computes some
        output.
    * Metrics: Computes any metrics that can be reported during training/testing.
    * Device: Handle pyTorch's annoying aspects about device placement and store which device this
        component is in.

    To implement your own component override:
        * _init_component_metadata (optional)
        * _load_metadata_from_sample (optional)
        * _finalize_component_metadata_and_model (optional)
        * load_data_from_sample (mandatory)
        * initialize_minibatch (mandatory)
        * extend_minibatch_by_sample (mandatory)
        * finalize_minibatch (mandatory)
        * _component_metrics (optional)
        * _reset_component_metrics (optional; mandatory if overriding `_component_metrics`)
    """

    def __init__(self, name: str, hyperparameters: Optional[Dict[str, Any]] = None):
        super(BaseComponent, self).__init__()
        self.__name = name
        self.__metadata_initialized = False
        self.__metadata_finalized = False
        self.__set_hyperparameters(hyperparameters)

    @property
    @final
    def name(self) -> str:
        return self.__name

    # region Metadata Loading
    @final
    def init_metadata(self) -> None:
        """
        Initialize metadata recursively for all children components, by invoking
        _init_component_metadata() for this component and all its children.
        """
        if not self.__metadata_initialized:
            # Initialize metadata for all children
            for child_module in self.children():
                if isinstance(child_module, BaseComponent):
                    child_module.init_metadata()

            self._init_component_metadata()
            self.__metadata_initialized = True

    def _init_component_metadata(self) -> None:
        """
        Initialize the component's metadata. This usually initializes a set of temporary objects that will be
        updated by `_load_metadata_from_sample` and converted to the final metadata by `_finalize_component_metadata_and_model`.

        For example, a component might create a token Counter at this stage, then update the counter from
        `load_metadata_from_sample` and create the vocabulary at `finalize_metadata`.
        """
        pass

    @final
    def load_metadata_from_sample(self, data_to_load: InputData) -> None:
        """
        Accept a single data point and load any metadata-related information from it.
        """
        assert self.__metadata_initialized, 'Metadata is not initialized'
        if not self.__metadata_finalized:
            self._load_metadata_from_sample(data_to_load)

    @abstractmethod
    def _load_metadata_from_sample(self, data_to_load: InputData) -> None:
        """
         Accept a single data point and load any metadata-related information from it.

        Implementors of this function should:
        * Load any metadata that are required by the component.
        * Unpack any parts of the parts of `data_to_load` that are needed for the children components
            and invoke their `load_metadata_from_sample` for those components.

        This means that for any child component the load_metadata_from_sample needs to be *explicitly* invoked.

        :param data_to_load: data relevant to this component and its children from a single data point.
        """
        pass

    @final
    def finalize_metadata_and_model(self) -> None:
        """
        Compute the final metadata that this component will be using.
        Recursively finalize the metadata for all children too.
        """
        if not self.__metadata_finalized:
            for child_module in self.children():
                if isinstance(child_module, BaseComponent):
                    child_module.finalize_metadata_and_model()

            self._finalize_component_metadata_and_model()
            self.__metadata_finalized = True

    def _finalize_component_metadata_and_model(self) -> None:
        """
        Finalize the metadata that this component will contain.

        Note to implementors: children component's metadata will have already been finalized when
            this function is called and thus they may be used.
        """
        pass

    # endregion

    # region Hyperparameters
    @classmethod
    @abstractmethod
    def default_hyperparameters(cls) -> Dict[str, Any]:
        """
        :return: the default hyperparameters of this component.
        """
        pass

    @final
    def __set_hyperparameters(self, component_hyperparameters: Optional[Dict[str, Any]]) -> None:
        """
        Set the component hyperparameters.
        """
        self.__hyperparameters_dict = self.default_hyperparameters()
        if component_hyperparameters is not None:
            self.__hyperparameters_dict.update(component_hyperparameters)

    @final
    def get_hyperparameter(self, name: str) -> Any:
        if name in self.__hyperparameters_dict:
            return self.__hyperparameters_dict[name]
        return self.default_hyperparameters()[name]

    @final
    @property
    def __hyperparameters(self) -> Dict[str, Any]:
        if not hasattr(self, '__hyperparameters_dict'):
            self.__hyperparameters_dict = self.default_hyperparameters()
        return self.__hyperparameters_dict

    @final
    @property
    def hyperparameters(self) -> Dict[str, Any]:
        """
        Get the hyperparameters of the component and its children.
        """
        hypers = {self.__name: dict(self.__hyperparameters_dict)}
        for child_module in self.children():
            if isinstance(child_module, BaseComponent):
                hypers[self.__name][child_module.__name] = child_module.hyperparameters
        return hypers

    # endregion

    # region Device Utilities
    @property
    def device(self):
        """Retrieve the device where this component lives."""
        return self.__device

    @final
    def to(self, *args, **kwargs):
        super(BaseComponent, self).to(*args, **kwargs)
        # Ugly but seemingly necessary hack: implicit dependency on non-exposed interface.
        device, _, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        self.__device = device
        for child_module in self.children():
            child_module.to(*args, **kwargs)

    @final
    def cuda(self, device=None):
        """Move the component to a GPU."""
        super(BaseComponent, self).cuda(device=device)
        self.__device = device or 'cuda:0'
        for child_module in self.children():
            child_module.cuda(device=device)

    @final
    def cpu(self):
        """Move the component to the CPU."""
        super(BaseComponent, self).cpu()
        self.__device = 'cpu'
        for child_module in self.children():
            child_module.cpu()

    # endregion

    # region Tensor Conversion
    @abstractmethod
    def load_data_from_sample(self, data_to_load: InputData) -> Optional[TensorizedData]:
        """
        This is called to load the data (tensorize) from a single example in a form that can be consumed by the
        neural network.

        Note to implementors: this usually involves unpacking data_to_load and invoking children component's
          load_data_from_sample so that each component loads parts of the data it cares about and then composing
          them into a single object along with any extra information.

        :param data_to_load: The data sample to be loaded.
        :return: A data structure that contains the tensorized data for the given sample
            or None if the datapoint should be rejected.
        """
        pass

    # endregion

    # region Minibatching Logic
    @abstractmethod
    def initialize_minibatch(self) -> Dict[str, Any]:
        """
        Initialize a dictionary that will be populated by `extend_minibatch_by_sample`.
        """
        pass

    @abstractmethod
    def extend_minibatch_by_sample(self, datapoint: TensorizedData, accumulated_minibatch_data: Dict[str, Any]) -> bool:
        """
        Add a datapoint to the minibatch. If for some component-related reason the minibatch cannot accumulate
            additional samples, this function should return False.

        :param datapoint: the datapoint to be added. This is a what `load_data_from_sample` returns.
        :param accumulated_minibatch_data: the minibatch data to be populated.
        :return true if we can continue extending the minibatch. False if for some reason the minibatch is full.
        """
        pass

    @abstractmethod
    def finalize_minibatch(self, accumulated_minibatch_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Finalize the minibatch data and make sure that the data is in an appropriate format to be consumed by
        the model. Commonly the values of the returned dictionary are `torch.tensor(..., device=device)`.

        :param accumulated_minibatch_data: the data that has been accumulated by `extend_minibatch_by_sample`.
        :return: the dictionary that will be passed as `**kwargs` to this component `forward()`
        """
        pass

    @final
    def create_minibatch(self, data_iterator_to_consume: Iterator[TensorizedData], max_num_items: int) -> \
            Tuple[Dict[str, Any], bool, int]:
        """
        Creates a minibatch from a finalized minibatch.

        :return: the data of the minibatch, a bool indicating whether the data iterator was fully consumed and
                the number of elements in the minibatch
        """
        mb_data = self.initialize_minibatch()
        num_elements_added = 0
        for element in islice(data_iterator_to_consume, max_num_items):
            continue_extending = self.extend_minibatch_by_sample(element, mb_data)
            num_elements_added += 1
            if not continue_extending:
                # The implementation of the component asked to stop extending the minibatch.
                batch_is_full = True
                break
        else:
            # At this point, the batch is full if we finished iterating through the loop and have max_num_items
            batch_is_full = num_elements_added == max_num_items

        if num_elements_added == 0:
            assert not batch_is_full, 'The data iterator was not exhausted but zero items were returned.'
            return {}, True, 0

        return self.finalize_minibatch(mb_data), batch_is_full, num_elements_added

    # endregion

    # region Component Loading/Unloading
    @final
    def save(self, path: RichPath) -> None:
        """Save the model at a given location."""
        with TemporaryDirectory() as tmpdir:
            target_file = os.path.join(tmpdir, 'model.pkl.gz')
            with gzip.open(target_file, 'wb') as f:
                torch.save(self, f)
            path.copy_from(RichPath.create(target_file))

    @classmethod
    def restore_model(cls, path: RichPath, device=None) -> 'BaseComponent':
        """Restore model to a given device."""
        model_path = path.to_local_path().path
        with gzip.open(model_path, 'rb') as f:
            model = torch.load(f, map_location=device)  # type: BaseComponent
        if device is not None:
            model.to(device)
        return model

    # endregion

    # region Model Statistics and Metrics
    @final
    def report_metrics(self) -> Dict[str, Any]:
        """
        Report the collected metrics for this component and its children.

        Each component can internally collect its own metrics as the implementor sees fit. For example,
        a counter may be incremented when the `forward()` function is invoked or a running average may
        by updated when a loss is computed. The metrics counter can be reset outside of the component
        when `reset_metrics` is invoked.

        To add metrics to a component, implementors need to:
        * Implement `_component_metrics` that computes the reported metrics from any component-internal variables.
        * Implement `_reset_component_metrics` which resets any variables that compute metrics.
        * Store any metric-related variables as fields in their component.

        """
        metrics = self._component_metrics()
        for child_module in self.children():
            if isinstance(child_module, BaseComponent):
                child_metrics = child_module._component_metrics()
                if len(child_metrics) > 0:
                    metrics[child_module.__name] = child_metrics
        return metrics

    @final
    def reset_metrics(self) -> None:
        """Reset any reported metrics. Often called after report_metrics() to reset any counters etc."""
        self._reset_component_metrics()
        for child_module in self.children():
            if isinstance(child_module, BaseComponent):
                child_module._reset_component_metrics()

    def _component_metrics(self) -> Dict[str, Any]:
        """
        Return a dictionary of metrics for the current component.

        The key is the name of the metric as it will be appear reported.
        The value can be anything, but using a formatted string may often be the preferred choice.
        """
        return {}

    def _reset_component_metrics(self) -> None:
        """Reset any metrics related to the component, such as any counters, running sums, averages, etc."""
        pass

    def num_parameters(self) -> int:
        """Compute the number of trainable parameters in this component and its children."""
        return sum(param.numel() for param in self.parameters(recurse=True) if param.requires_grad)
    # endregion
