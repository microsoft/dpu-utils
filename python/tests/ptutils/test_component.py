import unittest
import tempfile
from typing import Tuple, Iterator, Iterable

import torch

from dpu_utils.utils import RichPath
from dpu_utils.ptutils import ComponentTrainer

from tests.ptutils.testdata import SyntheticData
from tests.ptutils.testmodel import SimpleRegression, SampleDatapoint


class TestPytorchComponent(unittest.TestCase):
    def test_train_model(self):
        num_features = 100
        training_data, validation_data = self.__get_data(num_features)

        with tempfile.TemporaryDirectory() as dir:
            model_file = RichPath.create(dir).join('tmp.pkl.gz')

            model = SimpleRegression('SimpleRegressionTest', num_features)
            trainer = ComponentTrainer(model, model_file, max_num_epochs=50)
            trainer.train(training_data, validation_data, parallel_minibatch_creation=True)
            model_acc_1 = self.__compute_accuracy(model, validation_data)

            trained_model = SimpleRegression.restore_model(model_file)  # type: SimpleRegression
            trained_model_acc = self.__compute_accuracy(trained_model, validation_data)
            self.assertGreater(trained_model_acc, .95, f'Model achieves too low accuracy, {trained_model_acc:%}')

            self.assertAlmostEqual(trained_model_acc, model_acc_1, places=3, msg=f'Accuracy before and after loading does not match: {trained_model_acc} vs {model_acc_1}')

    def test_freeze_params(self):
        num_features = 100
        training_data, validation_data = self.__get_data(num_features)

        with tempfile.TemporaryDirectory() as dir:
            model_file = RichPath.create(dir).join('tmp.pkl.gz')

            model = SimpleRegression('SimpleRegressionTest', num_features)
            trainer = ComponentTrainer(model, model_file, max_num_epochs=50)

            def get_freeze_weights():
                for p in model.parameters():
                    if len(p.shape) == 2:  # Just the weights
                        yield p

            trainer.train(training_data, validation_data, get_parameters_to_freeze=lambda: set(get_freeze_weights()))
            trained_model_acc = self.__compute_accuracy(model, validation_data)

            self.assertLess(trained_model_acc, .7, f'Model achieves too high accuracy but the weights were frozen, {trained_model_acc:%}')


    def __get_data(self, num_features):
        data = SyntheticData(num_features)
        all_data = list(data.generate(10000))
        training_data, validation_data = all_data[:9000], all_data[9000:]
        return training_data, validation_data

    def __compute_accuracy(self, model: SimpleRegression, dataset: Iterable[SampleDatapoint]) -> float:
        num_samples = 0
        num_correct = 0
        for point, prediction in self.__get_model_prediction(model, dataset):
            num_samples += 1
            if point.target_class == prediction:
                num_correct += 1
        return num_correct / num_samples

    def __get_model_prediction(self, model: SimpleRegression, data: Iterable[SampleDatapoint]) -> Iterator[Tuple[SampleDatapoint, bool]]:
        for datapoint in data:
            tensorized = model.load_data_from_sample(datapoint)
            mb_data = model.initialize_minibatch()
            model.extend_minibatch_by_sample(tensorized, mb_data)
            mb_data = model.finalize_minibatch(mb_data)

            with torch.no_grad():
                predictions = model.predict(mb_data['inputs']).cpu().numpy()
            yield datapoint, predictions[0]



if __name__ == '__main__':
    unittest.main()
