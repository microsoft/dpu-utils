import unittest

from dpu_utils.utils.iterators import shuffled_iterator


class TestShuffleIterator(unittest.TestCase):

    def test_return_all_elements(self):
        for size in [100, 10000, 100000]:
            shuffled_dataset = list(shuffled_iterator(range(size)))
            self.assertNotEqual(shuffled_dataset[:10], list(range(10)),
                                'It is highly unlikely that the original order is preserved')
            self.assertSetEqual(set(shuffled_dataset), set(range(size)), f'Some returned elements are missing.')

