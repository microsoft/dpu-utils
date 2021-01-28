import json
import logging
import time
from abc import ABC, abstractmethod
import math
from typing import Optional, Iterable, Set, TypeVar, Generic, Callable, List, Dict, Iterator, Tuple, Union, Any

import torch
from tqdm import tqdm

from .basecomponent import BaseComponent
from dpu_utils.utils import RichPath, ThreadedIterator
from dpu_utils.utils.iterators import shuffled_iterator

InputData = TypeVar('InputData')
TensorizedData = TypeVar('TensorizedData')
EndOfEpochHook = Callable[[BaseComponent, int, Dict], None]

__all__ = ['ComponentTrainer', 'AbstractScheduler']

class AbstractScheduler(ABC):
    @abstractmethod
    def step(self, epoch_idx: int, epoch_step: int)-> None:
        pass

class ComponentTrainer(Generic[InputData, TensorizedData]):
    """
    A trainer for `BaseComponent`s. Used mainly for supervised learning.

    Create a `ComponentTrainer` by passing a `BaseComponent` in the constructor.
    Invoke `train()` to initiate the training loop.

    """

    LOGGER = logging.getLogger('ComponentTrainer')

    def __init__(self, model: BaseComponent[InputData, TensorizedData], save_location: RichPath,
                 *, max_num_epochs: int = 200, minibatch_size: int = 200,
                 optimizer_creator: Optional[Callable[[Iterable[torch.Tensor]], torch.optim.Optimizer]]=None,
                 scheduler_creator: Optional[Callable[[torch.optim.Optimizer], AbstractScheduler]]=None):
        """

        :param model: The model to be trained.
        :param save_location: The location where the trained model will be checkpointed and saved.
        :param max_num_epochs: The maximum number of epochs to run training for.
        :param minibatch_size: The maximum size of the minibatch (`BaseComponent`s can override this
            by detecting full minibatches and returning False in `extend_minibatch_by_sample`)
        :param optimizer_creator: An optional function that accepts an iterable of the training parameters
            (pyTorch tensors) and returns a PyTorch optimizer.
        :param scheduler_creator: An optional function that accepts an optimizer and creates a scheduler
            implementing `AbstractScheduler`. This could be a wrapper for existing learning schedulers.
            The scheduler will be invoked at after each training step.
        """
        self.__model = model
        self.__save_location = save_location
        assert save_location.path.endswith('.pkl.gz'), 'All models are stored as .pkl.gz. Please indicate this in the save_location.'

        self.__max_num_epochs = max_num_epochs
        self.__minibatch_size = minibatch_size
        if optimizer_creator is None:
            self.__create_optimizer = torch.optim.Adam
        else:
            self.__create_optimizer = optimizer_creator

        self.__create_scheduler = scheduler_creator

        self.__train_epoch_end_hooks = []  # type: List[EndOfEpochHook]
        self.__validation_epoch_end_hooks = [] # type: List[EndOfEpochHook]
        self.__metadata_finalized_hooks = []  # type: List[Callable[[BaseComponent], None]]
        self.__training_start_hooks = []   # type: List[Callable[[BaseComponent, torch.optim.Optimizer], None]]

    @property
    def model(self) -> BaseComponent[InputData, TensorizedData]:
        return self.__model

    def __load_metadata(self, training_data: Iterable[InputData]) -> None:
        """
        Ask all components of the model to compute their metadata by doing a full pass over the training data.
        """
        self.__model.init_metadata()
        for element in training_data:
            self.__model.load_metadata_from_sample(element)
        self.__model.finalize_metadata_and_model()
        self.LOGGER.info('Model metadata loaded. The following model was created:\n %s', self.__model)
        self.LOGGER.info('Hyperparameters:\n %s', json.dumps(self.__model.hyperparameters, indent=2))
        for hook in self.__metadata_finalized_hooks:
            hook(self.__model)

    def __save_current_model(self) -> None:
        self.__model.save(self.__save_location)

    def restore_model(self, device=None) -> None:
        self.__model = BaseComponent.restore_model(self.__save_location, device=device)

    def register_train_epoch_end_hook(self, hook: EndOfEpochHook) -> None:
        self.__train_epoch_end_hooks.append(hook)

    def register_validation_epoch_end_hook(self, hook: EndOfEpochHook) -> None:
        self.__validation_epoch_end_hooks.append(hook)

    def register_model_metadata_finalized_hook(self, hook: Callable[[BaseComponent], None]) -> None:
        self.__metadata_finalized_hooks.append(hook)

    def register_training_start_hook(self, hook: Callable[[BaseComponent, torch.optim.Optimizer], None]) -> None:
        self.__training_start_hooks.append(hook)

    def train(self, training_data: Iterable[InputData], validation_data: Iterable[InputData],
              show_progress_bar: bool = True, patience: int = 5, initialize_metadata: bool = True,
              exponential_running_average_factor: float = 0.97, get_parameters_to_freeze: Optional[Callable[[], Set]] = None,
              parallel_minibatch_creation: bool=False, device: Optional[Union[str, torch.device]] = None) -> None:
        """
        The training-validation loop for `BaseComponent`s.

        :param training_data: An iterable that each iteration yields the full training data.
        :param validation_data: An iterable that each iteration yields the full validation data.
        :param show_progress_bar: Show a progress bar
        :param patience: The number of iterations before early stopping kicks in.
        :param initialize_metadata: If true, initialize the metadata from the training_data. Otherwise,
            assume that the model that is being trained has its metadata already initialized.
        :param exponential_running_average_factor: The factor of the running average of the training loss
            displayed in the progress bar.
        :param get_parameters_to_freeze: The (optional) callable that returns the set of parameters to freeze during training.
        :param parallel_minibatch_creation: If True the minibatches will be created in a separate thread.
        """
        if initialize_metadata:
            self.__load_metadata(training_data)

        self.LOGGER.info('Model has %s parameters', self.__model.num_parameters())
        self.LOGGER.debug('Data Tensorization Started...')

        def data_to_tensor_iterator(data):
            for datapoint in data:
                tensorized_datapoint = self.__model.load_data_from_sample(datapoint)
                if tensorized_datapoint is not None:
                    yield tensorized_datapoint

        def training_tensors():
            yield from ThreadedIterator(
                original_iterator=data_to_tensor_iterator(training_data),
                max_queue_size=10 * self.__minibatch_size
            )

        def validation_tensors():
            yield from ThreadedIterator(
                original_iterator=data_to_tensor_iterator(validation_data),
                max_queue_size=10 * self.__minibatch_size
            )

        def minibatch_iterator(data_iterator: Iterator[TensorizedData], return_partial_minibatches: bool = False) -> Tuple[Dict, int]:
            while True:
                mb_data, batch_is_full, num_elements = self.__model.create_minibatch(data_iterator, max_num_items=self.__minibatch_size)
                if num_elements == 0:
                    break
                elif not batch_is_full and not return_partial_minibatches:
                    break  # Do not return partial minibatches when the iterator is exhausted.
                else:
                    yield mb_data, num_elements

        if device is None:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.__model.to(device)
        self.LOGGER.info('Using %s for training.' % device)


        if get_parameters_to_freeze is None:
            get_parameters_to_freeze = lambda: set()
        trainable_parameters = set(self.__model.parameters()) - get_parameters_to_freeze()
        optimizer = self.__create_optimizer(trainable_parameters)
        scheduler = None if self.__create_scheduler is None else self.__create_scheduler(optimizer)

        for hook in self.__training_start_hooks:
            hook(self.__model, optimizer)

        best_loss = float('inf')  # type: float
        num_epochs_not_improved = 0  # type: int
        for epoch in range(self.__max_num_epochs):
            self.__model.train()

            data_iter = shuffled_iterator(training_tensors())
            sum_epoch_loss = 0.0
            running_avg_loss = 0.0
            num_minibatches = 0
            num_samples = 0

            start_time = time.time()
            self.__model.reset_metrics()
            with tqdm(desc='Training', disable=not show_progress_bar, leave=False) as progress_bar:
                for step_idx, (mb_data, num_elements) in enumerate(ThreadedIterator(
                        minibatch_iterator(data_iter, return_partial_minibatches=False),
                        enabled=parallel_minibatch_creation)):
                    optimizer.zero_grad()
                    mb_loss = self.__model(**mb_data)
                    mb_loss.backward()

                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step(epoch_idx=epoch, epoch_step=step_idx)

                    loss = float(mb_loss.cpu())
                    if math.isnan(loss):
                        raise Exception('Training Loss has a NaN value.')

                    sum_epoch_loss += loss
                    num_minibatches += 1
                    num_samples += num_elements

                    if num_minibatches == 1:  # First minibatch
                        running_avg_loss = loss
                    else:
                        running_avg_loss = exponential_running_average_factor * running_avg_loss + (
                                    1 - exponential_running_average_factor) * loss
                    progress_bar.update()
                    progress_bar.set_postfix(Loss=f'{running_avg_loss:.2f}')

            elapsed_time = time.time() - start_time  # type: float
            self.LOGGER.info('Training complete in %.1fsec [%.2f samples/sec]', elapsed_time,
                             (num_samples / elapsed_time))
            assert num_minibatches > 0, 'No training minibatches were created. The minibatch size may be too large or the training dataset size too small.'
            self.LOGGER.info('Epoch %i: Avg Train Loss %.2f', epoch + 1, sum_epoch_loss / num_minibatches)
            train_metrics = self.__model.report_metrics()
            for epoch_hook in self.__train_epoch_end_hooks:
                epoch_hook(self.__model, epoch, train_metrics)
            if len(train_metrics) > 0:
                self.LOGGER.info('Training Metrics: %s', json.dumps(train_metrics, indent=2))

            # Now do validation!
            self.__model.eval()
            data_iter = validation_tensors()
            sum_epoch_loss = 0
            num_minibatches = 0
            num_samples = 0
            start_time = time.time()
            self.__model.reset_metrics()
            with tqdm(desc='Validation', disable=not show_progress_bar, leave=False) as progress_bar, torch.no_grad():
                for mb_data, num_elements in ThreadedIterator(
                        minibatch_iterator(data_iter, return_partial_minibatches=True),
                        enabled=parallel_minibatch_creation):
                    mb_loss = self.__model(**mb_data)

                    loss = float(mb_loss.cpu())
                    if math.isnan(loss):
                        raise Exception('Validation Loss has a NaN value.')

                    sum_epoch_loss += loss
                    num_minibatches += 1
                    num_samples += num_elements

                    progress_bar.update()
                    progress_bar.set_postfix(Loss=f'{sum_epoch_loss / num_minibatches:.2f}')

            elapsed_time = time.time() - start_time
            assert num_samples > 0, 'No validation data was found.'
            validation_loss = sum_epoch_loss / num_minibatches
            self.LOGGER.info('Validation complete in %.1fsec [%.2f samples/sec]', elapsed_time,
                             (num_samples / elapsed_time))
            self.LOGGER.info('Epoch %i: Avg Valid Loss %.2f', epoch + 1, validation_loss)
            validation_metrics = self.__model.report_metrics()
            for epoch_hook in self.__validation_epoch_end_hooks:
                epoch_hook(self.__model, epoch, validation_metrics)
            if len(validation_metrics) > 0:
                self.LOGGER.info('Validation Metrics: %s', json.dumps(validation_metrics, indent=2))

            if validation_loss < best_loss:
                self.LOGGER.info('Best loss so far --- Saving model.')
                num_epochs_not_improved = 0
                self.__save_current_model()
                best_loss = validation_loss
            else:
                num_epochs_not_improved += 1
                if num_epochs_not_improved > patience:
                    self.LOGGER.warning('After %s epochs loss has not improved. Stopping.', num_epochs_not_improved)
                    break


        # Restore the best model that was found.
        self.restore_model()
