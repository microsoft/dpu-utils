import json
import logging
import time
from typing import Optional, Iterable, Set, TypeVar, Generic

import torch
from tqdm import tqdm

from .basecomponent import BaseComponent
from dpu_utils.utils import RichPath, ThreadedIterator
from dpu_utils.utils.iterators import shuffled_iterator

InputData = TypeVar('InputData')
TensorizedData = TypeVar('TensorizedData')

__all__ = ['ComponentTrainer']

class ComponentTrainer(Generic[InputData, TensorizedData]):
    """A trainer for `BaseComponent`s. Used mainly for supervised learning."""

    LOGGER = logging.getLogger('ComponentTrainer')

    def __init__(self, model: BaseComponent[InputData, TensorizedData], save_location: RichPath,
                 *, max_num_epochs: int = 200, minibatch_size: int = 200):
        self.__model = model
        self.__save_location = save_location
        assert save_location.path.endswith('.pkl.gz'), 'All models are stored as .pkl.gz. Please indicate this in the save_location.'

        self.__max_num_epochs = max_num_epochs
        self.__minibatch_size = minibatch_size

        # TODO: Use hypers to add different optimizers and their hyperparameters

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

    def __save_current_model(self) -> None:
        self.__model.save(self.__save_location)

    def restore_model(self, device=None) -> None:
        self.__model = BaseComponent.restore_model(self.__save_location, device=device)

    def train(self, training_data: Iterable[InputData], validation_data: Iterable[InputData],
              show_progress_bar: bool = True, patience: int = 5, initialize_metadata: bool = True,
              exponential_running_average_factor: float = 0.97, parameters_to_freeze: Optional[Set] = None):
        if initialize_metadata:
            self.__load_metadata(training_data)

        self.LOGGER.info('Model has %s parameters', self.__model.num_parameters())
        self.LOGGER.debug('Data Tensorization Started...')

        def training_tensors():
            yield from ThreadedIterator(
                original_iterator=(self.__model.load_data_from_sample(d) for d in training_data),
                max_queue_size=10 * self.__minibatch_size
            )

        def validation_tensors():
            yield from ThreadedIterator(
                original_iterator=(self.__model.load_data_from_sample(d) for d in validation_data),
                max_queue_size=10 * self.__minibatch_size
            )

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.LOGGER.info('Using %s for training.' % device)
        if torch.cuda.is_available():
            self.__model.cuda()
        else:
            self.__model.cpu()

        if parameters_to_freeze is None:
            parameters_to_freeze = set()
        optimizer = torch.optim.Adam(set(self.__model.parameters()) - parameters_to_freeze)

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
                while True:
                    mb_data, data_iterator_exhausted, num_elements = self.__model.create_minibatch(data_iter,
                                                                                          max_num_items=self.__minibatch_size)
                    if data_iterator_exhausted or num_elements == 0:
                        break  # Do not consider half-full or empty minibatches
                    optimizer.zero_grad()
                    mb_loss = self.__model.forward(**mb_data)
                    mb_loss.backward()

                    optimizer.step()
                    num_minibatches += 1
                    num_samples += num_elements
                    sum_epoch_loss += float(mb_loss.cpu())
                    if num_minibatches == 1:  # First minibatch
                        running_avg_loss = float(mb_loss.cpu())
                    else:
                        running_avg_loss = exponential_running_average_factor * running_avg_loss + (
                                    1 - exponential_running_average_factor) * float(mb_loss.cpu())
                    progress_bar.update()
                    progress_bar.set_postfix(Loss=f'{running_avg_loss:.2f}')

            elapsed_time = time.time() - start_time  # type: float
            self.LOGGER.info('Training complete in %.1fsec [%.2f samples/sec]', elapsed_time,
                             (num_samples / elapsed_time))
            self.LOGGER.info('Epoch %i: Avg Train Loss %.2f', epoch + 1, sum_epoch_loss / num_minibatches)
            train_metrics = self.__model.report_metrics()
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
                while True:
                    mb_data, data_iterator_exhausted, num_elements = self.__model.create_minibatch(data_iter,
                                                                                          max_num_items=self.__minibatch_size)
                    if num_elements == 0:
                        break  # No more elements could be found in the data_iter.
                    mb_loss = self.__model.forward(**mb_data)
                    num_minibatches += 1
                    num_samples += num_elements
                    sum_epoch_loss += float(mb_loss.cpu())
                    progress_bar.update()
                    progress_bar.set_postfix(Loss=f'{sum_epoch_loss / num_minibatches:.2f}')
                    if data_iterator_exhausted:
                        break  # No more elements in the data iterator

            elapsed_time = time.time() - start_time  # type: float
            validation_loss = sum_epoch_loss / num_minibatches
            self.LOGGER.info('Validation complete in %.1fsec [%.2f samples/sec]', elapsed_time,
                             (num_samples / elapsed_time))
            self.LOGGER.info('Epoch %i: Avg Valid Loss %.2f', epoch + 1, validation_loss)
            validation_metrics = self.__model.report_metrics()
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
