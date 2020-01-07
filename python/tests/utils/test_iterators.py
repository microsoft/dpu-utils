import unittest

from dpu_utils.utils import shuffled_iterator, ThreadedIterator, BufferedIterator, DoubleBufferedIterator


class TestShuffleIterator(unittest.TestCase):

    def test_return_all_elements(self):
        for size in [100, 10000, 100000]:
            shuffled_dataset = list(shuffled_iterator(range(size)))
            self.assertNotEqual(shuffled_dataset[:10], list(range(10)),
                                'It is highly unlikely that the original order is preserved')
            self.assertSetEqual(set(shuffled_dataset), set(range(size)), f'Some returned elements are missing.')


class TestParellelIterators(unittest.TestCase):

    ALL_ITERATOR_TYPES = [ThreadedIterator, BufferedIterator, DoubleBufferedIterator]

    def test_return_all_elements_in_order(self):
        for iterator_type in self.ALL_ITERATOR_TYPES:
            for size in [100, 10000]:
                returned = list(iterator_type(range(size)))
                self.assertListEqual(returned, list(range(size)), f'Iterator {iterator_type.__name__} did not return all elements.')