import multiprocessing
import sys
import queue
import threading
import traceback
from typing import TypeVar, Iterable, Iterator, List, Callable, Optional

T = TypeVar('T')

__all__ = ['ThreadedIterator', 'MultiWorkerCallableIterator', 'BufferedIterator', 'DoubleBufferedIterator']


class ThreadedIterator(Iterator[T]):
    """An iterator object that computes its elements in a single parallel thread to be ready to be consumed.
    The iterator should *not* return `None`. Elements of the original iterable will be shuffled arbitrarily."""
    def __init__(self, original_iterator: Iterator[T], max_queue_size: int = 2, enabled: bool = True):
        self.__is_enabled = enabled
        if enabled:
            self.__queue = queue.Queue(maxsize=max_queue_size)  # type: queue.Queue[Optional[T]]
            self.__thread = threading.Thread(target=lambda: self.__worker(self.__queue, original_iterator))
            self.__thread.start()
        else:
            self.__original_iterator = original_iterator

    @staticmethod
    def __worker(queue: queue.Queue, original_iterator: Iterator[T])-> None:
        try:
            for element in original_iterator:
                assert element is not None, 'By convention, Iterables wrapped in ThreadedIterator may not contain None.'
                queue.put(element, block=True)
            queue.put(None, block=True)
        except Exception as e:
            _, __, tb = sys.exc_info()
            queue.put((e, tb), block=True)

    def __next__(self) -> T:
        next_element = self.__queue.get(block=True)
        if next_element is None:
            self.__thread.join()
            self.__queue.put(None)  # Make sure that we remember that we are done if we are called once more...
            raise StopIteration
        if isinstance(next_element, tuple) and isinstance(next_element[0], Exception):
            raise next_element[0].with_traceback(next_element[1])
        return next_element

    def __iter__(self):
        if self.__is_enabled:
            return self
        else:
            return self.__original_iterator


class MultiWorkerCallableIterator(Iterable):
    """An iterator that computes its elements in parallel workers to be ready to be consumed. The iterator should
    have at least one element. The order of the callables is shuffled arbitrarily."""

    def __init__(self, argument_iterator: Iterator[Iterable], worker_callable: Callable, max_queue_size: int=1, num_workers: int = 5, use_threads: bool=True):
        self.__in_queue = queue.Queue() if use_threads else multiprocessing.Queue()
        self.__num_elements = 0
        for callable_args in argument_iterator:
            self.__in_queue.put(callable_args)
            self.__num_elements += 1
        self.__out_queue = queue.Queue(maxsize=max_queue_size) if use_threads else multiprocessing.Queue(
            maxsize=max_queue_size
        )
        self.__threads = [
            threading.Thread(target=lambda: self.__worker(worker_callable)) if use_threads
            else multiprocessing.Process(target=lambda: self.__worker(worker_callable)) for _ in range(num_workers)
        ]
        for worker in self.__threads:
            worker.start()

    def __worker(self, worker_callable):
        try:
            while not self.__in_queue.empty():
                next_element = self.__in_queue.get(block=False)
                result = worker_callable(*next_element)
                self.__out_queue.put(result)
        except queue.Empty:
            pass
        except Exception as e:
            _, __, tb = sys.exc_info()
            self.__out_queue.put((e, tb), block=True)

    def __iter__(self):
        for _ in range(self.__num_elements):
            next_element = self.__out_queue.get(block=True)
            if isinstance(next_element, tuple) and isinstance(next_element[0], Exception):
                raise next_element[0].with_traceback(next_element[1])
            yield next_element

        for worker in self.__threads:
            worker.join()


class BufferedIterator(Iterable[T]):
    """An iterator object that computes its elements in a parallel process, ready to be consumed.
    The iterator should *not* return None"""

    def __init__(self, original_iterator: Iterator[T], max_queue_size: int=3, enabled: bool=True):
        self.__original_iterator = original_iterator
        self.__is_enabled = enabled

        if enabled:
            self.__buffer = multiprocessing.Queue(maxsize=max_queue_size)
            self.__worker_process = multiprocessing.Process(target=lambda: self.__worker(original_iterator))
            self.__worker_process.start()

    def __worker(self, original_iterator: Iterator[T]) -> None:
        """Implementation of worker thread. Iterates over the original iterator, pulling results
        and putting them into a buffer."""
        try:
            for element in original_iterator:
                assert element is not None, 'By convention, iterator elements must not be None'
                self.__buffer.put(element, block=True)
            self.__buffer.put(None, block=True)
        except Exception as e:
            _, __, tb = sys.exc_info()
            self.__buffer.put((e, tb), block=True)

    def __iter__(self):
        if not self.__is_enabled:
            yield from self.__original_iterator
            return

        next_element = self.__buffer.get(block=True)
        while next_element is not None:
            if isinstance(next_element, tuple) and isinstance(next_element[0], Exception):
                raise next_element[0].with_traceback(next_element[1])
            yield next_element
            next_element = self.__buffer.get(block=True)

        self.__worker_process.join()

class DoubleBufferedIterator(Iterator[T]):
    """An iterator object that wraps double buffering around an iterable sequence.
    This avoids waits in downstream applications if each step of the inner iterable can take a long while,
    as the Queue used in (Single)BufferedIterator requires consumer and producer to synchronize.

    Note: The inner iterable should *not* return None"""

    def __init__(self, original_iterable: Iterable[T], max_queue_size_inner: int=20, max_queue_size_outer: int=5):
        self.__buffer_inner = multiprocessing.Queue(maxsize=max_queue_size_inner)
        self.__buffer_outer = multiprocessing.Queue(maxsize=max_queue_size_outer)
        self.__worker_process_inner = multiprocessing.Process(target=lambda: self.__worker_inner(original_iterable))
        self.__worker_process_outer = multiprocessing.Process(target=lambda: self.__worker_outer())
        self.__worker_process_inner.start()
        self.__worker_process_outer.start()

    def __worker_inner(self, original_iterator: Iterable[T]) -> None:
        """Consumes elements from the original iterator, putting them into an inner buffer."""
        try:
            for element in original_iterator:
                assert element is not None, 'By convention, iterator elements must not be None'
                self.__buffer_inner.put(element, block=True)
            self.__buffer_inner.put(None, block=True)
        except Exception as e:
            _, __, tb = sys.exc_info()
            print("!!! Exception '%s' in inner worker of DoubleBufferedIterator:\n %s" % (e, "".join(
                traceback.format_tb(tb)
            )))
            self.__buffer_inner.put((e, tb), block=True)

    def __worker_outer(self) -> None:
        """Consumes elements from the inner worker and just passes them through to the outer buffer."""
        try:
            next_element = self.__buffer_inner.get(block=True)
            while next_element is not None:
                self.__buffer_outer.put(next_element, block=True)
                next_element = self.__buffer_inner.get(block=True)
            self.__buffer_outer.put(next_element, block=True)
        except Exception as e:
            _, __, tb = sys.exc_info()
            print("!!! Exception '%s' in outer worker of DoubleBufferedIterator:\n %s" % (
                e, "".join(traceback.format_tb(tb))
            ))
            self.__buffer_outer.put((e, tb), block=True)

    def __iter__(self):
        return self

    def __next__(self):
        next_element = self.__buffer_outer.get(block=True)
        if isinstance(next_element, tuple) and isinstance(next_element[0], Exception):
            raise next_element[0].with_traceback(next_element[1])
        elif next_element is None:
            self.__worker_process_inner.join()
            self.__worker_process_outer.join()
            raise StopIteration
        return next_element
