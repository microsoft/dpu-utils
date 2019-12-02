import unittest

from dpu_utils.utils.iterators import shuffled_iterator


class TestShuffleIterator(unittest.TestCase):

    def test_return_all(self):
        for size in [10, 10000, 100000]:
            shuffled_dataset = set(shuffled_iterator(range(size)))
            self.assertSetEqual(shuffled_dataset, set(range(size)), f'Some returned elements are missing.')

