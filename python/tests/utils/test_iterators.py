import unittest
from functools import partial
from itertools import islice

from dpu_utils.utils import shuffled_iterator, ThreadedIterator, BufferedIterator, DoubleBufferedIterator, MultiWorkerCallableIterator


class TestShuffleIterator(unittest.TestCase):

    def test_return_all_elements(self):
        for size in [100, 10000, 100000]:
            shuffled_dataset = list(shuffled_iterator(range(size)))
            self.assertNotEqual(shuffled_dataset[:10], list(range(10)),
                                'It is highly unlikely that the original order is preserved')
            self.assertSetEqual(set(shuffled_dataset), set(range(size)), f'Some returned elements are missing.')

def identity(x):
    return x

class TestMultiWorkerIterator(unittest.TestCase):
    def test_return_all_elements(self):
        for use_threads in (True, False):
            with self.subTest('useThread={%s}' % use_threads):
                for size in [100, 10000, 100000]:
                    dataset = list(MultiWorkerCallableIterator(((i,) for i in range(size)), identity, use_threads=use_threads))
                    self.assertSetEqual(set(dataset), set(range(size)), f'Some returned elements are missing.')


def generator(size):
    for i in range(size):
        yield i

class IterWrapper:
    def __init__(self, iter_fn):
        self._iter_fn = iter_fn

    def __iter__(self):
        yield from self._iter_fn()

class TestParellelIterators(unittest.TestCase):

    ALL_ITERATOR_TYPES = [ThreadedIterator, BufferedIterator, DoubleBufferedIterator]

    def test_return_all_elements_in_order(self):
        for iterator_type in self.ALL_ITERATOR_TYPES:
            for enabled in (True, False):
                for size in [100, 10000]:
                    for iter_kind in (range(size), generator(size), IterWrapper(partial(generator, size))):
                        with self.subTest("%s-%s-%s-enabled=%s" % (iterator_type, size, iter_kind, enabled)):
                            returned = list(iterator_type(iter_kind, enabled=enabled))
                            self.assertListEqual(returned, list(range(size)), f'Iterator {iterator_type.__name__} did not return all elements.')


    def test_finish_on_partial_iteration(self):
        """Parallel iterators may leave resources (threads, processes) on partial iteration. Ensure that's not the case."""
        for iterator_type in self.ALL_ITERATOR_TYPES:
            for iter_kind in (range(100), generator(100), IterWrapper(partial(generator, 100))):
                with self.subTest("%s=%s" % (iterator_type, iter_kind)):
                    returned = list(islice(iterator_type(iter_kind), 10))
        # The test always finishes normally, but the pytest process should _not_ hang due to unfinished threads/processes.