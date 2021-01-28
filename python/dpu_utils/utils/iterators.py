import multiprocessing
from functools import partial
import random
import sys
import queue
import threading
import traceback
from itertools import islice
from typing import Any, TypeVar, Iterable, Iterator, List, Callable, Optional, Union, Tuple

T = TypeVar('T')

__all__ = ['ThreadedIterator', 'MultiWorkerCallableIterator', 'BufferedIterator', 'DoubleBufferedIterator', 'shuffled_iterator']


class ThreadedIterator(Iterator[T]):
    """An iterator object that computes its elements in a single parallel thread to be ready to be consumed.
    The iterator should *not* return `None`. Elements of the original iterable will be shuffled arbitrarily."""
    def __init__(self, original_iterator: Iterator[T], max_queue_size: int = 2, enabled: bool = True):
        self.__is_enabled = enabled
        if enabled:
            self.__queue = queue.Queue(maxsize=max_queue_size)  # type: queue.Queue[Optional[T]]
            self.__thread = threading.Thread(target=lambda: self.__worker(self.__queue, original_iterator), daemon=True)
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
        self.__in_queue = queue.Queue() if use_threads else multiprocessing.Queue()  # type: Union[queue.Queue, multiprocessing.Queue]
        self.__num_elements = 0
        for callable_args in argument_iterator:
            self.__in_queue.put(callable_args)
            self.__num_elements += 1
        self.__out_queue = queue.Queue(maxsize=max_queue_size) if use_threads else multiprocessing.Queue(
            maxsize=max_queue_size
        ) # type: Union[queue.Queue, multiprocessing.Queue]
        self.__threads = [
            threading.Thread(target=lambda: self.__worker(worker_callable)) if use_threads
            else multiprocessing.Process(target=lambda: self.__worker(worker_callable)) for _ in range(num_workers)
        ]  # type: List[Union[threading.Thread, multiprocessing.Process]]
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
            self.__buffer = multiprocessing.Queue(maxsize=max_queue_size)  # type: multiprocessing.Queue[Union[None, T, Tuple[Exception, Any]]]
            self.__worker_process = multiprocessing.Process(target=partial(BufferedIterator._worker, self.__buffer, original_iterator))
            self.__worker_process.start()

    @staticmethod
    def _worker(buffer, original_iterator: Iterator[T]) -> None:
        """Implementation of worker thread. Iterates over the original iterator, pulling results
        and putting them into a buffer."""
        try:
            for element in original_iterator:
                assert element is not None, 'By convention, iterator elements must not be None'
                buffer.put(element, block=True)
            buffer.put(None, block=True)
        except Exception as e:
            _, __, tb = sys.exc_info()
            buffer.put((e, tb), block=True)

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
        self.__buffer_inner = multiprocessing.Queue(maxsize=max_queue_size_inner)  # type: multiprocessing.Queue[Union[None, T, Tuple[Exception, Any]]]
        self.__buffer_outer = multiprocessing.Queue(maxsize=max_queue_size_outer)  # type: multiprocessing.Queue[Union[None, T, Tuple[Exception, Any]]]
        self.__worker_process_inner = multiprocessing.Process(target=partial(DoubleBufferedIterator._worker_inner, self.__buffer_inner, original_iterable))
        self.__worker_process_outer = multiprocessing.Process(target=partial(DoubleBufferedIterator._worker_outer, self.__buffer_inner, self.__buffer_outer))
        self.__worker_process_inner.start()
        self.__worker_process_outer.start()

    @staticmethod
    def _worker_inner(buffer_inner, original_iterator: Iterable[T]) -> None:
        """Consumes elements from the original iterator, putting them into an inner buffer."""
        try:
            for element in original_iterator:
                assert element is not None, 'By convention, iterator elements must not be None'
                buffer_inner.put(element, block=True)
            buffer_inner.put(None, block=True)
        except Exception as e:
            _, __, tb = sys.exc_info()
            print("!!! Exception '%s' in inner worker of DoubleBufferedIterator:\n %s" % (e, "".join(
                traceback.format_tb(tb)
            )))
            buffer_inner.put((e, tb), block=True)

    @staticmethod
    def _worker_outer(buffer_inner, buffer_outer) -> None:
        """Consumes elements from the inner worker and just passes them through to the outer buffer."""
        try:
            next_element = buffer_inner.get(block=True)
            while next_element is not None:
                buffer_outer.put(next_element, block=True)
                next_element = buffer_inner.get(block=True)
            buffer_outer.put(next_element, block=True)
        except Exception as e:
            _, __, tb = sys.exc_info()
            print("!!! Exception '%s' in outer worker of DoubleBufferedIterator:\n %s" % (
                e, "".join(traceback.format_tb(tb))
            ))
            buffer_outer.put((e, tb), block=True)

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


def shuffled_iterator(input_iterator: Iterator[T], buffer_size: int = 10000, rng: Optional[random.Random]=None) -> Iterator[T]:
    """
    Accept an iterator and return an approximate streaming (and memory efficient) shuffled iterator.

    To achieve (approximate) shuffling a buffer of elements is stored. Once the buffer is full, it is shuffled
    and random elements are yielded from the buffer, while it continues to be replenished.

    Notes:
         * There is a good bias for yielding the first set of elements in input early.
         * There is a delay for this wrapper to yield elements as the buffer needs to be filled in first
            with `buffer_size` elements or the `input_iterator` to be exhausted.

    """
    if rng is None:
        rng = random
    # Ensure that this is an iterator that can be consumed exactly once.
    input_iterator = iter(input_iterator)

    buffer = list(islice(input_iterator, buffer_size))  # type: List[T]
    rng.shuffle(buffer)

    for element in input_iterator:
        # Pick a random element in the buffer to yield and replace it with a new element
        idx = rng.randrange(buffer_size)
        to_yield, buffer[idx] = buffer[idx], element
        yield to_yield

    yield from buffer
