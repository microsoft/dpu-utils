from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Iterable, TypeVar, Generic, Union

from dpu_utils.utils import RichPath

T = TypeVar('T')

__all__ = ['ChunkWriter']


class ChunkWriter(Generic[T]):
    """Encapsulates writing output into chunks. By setting the file_suffix to either .pkl.gz or .json.gz
    the appropriate format will be used for the chunks."""
    def __init__(self, out_folder: Union[RichPath, str], file_prefix: str, max_chunk_size: int, file_suffix: str,
                 parallel_writers: int=0):
        self.__current_chunk = []  # type: List[T]
        if isinstance(out_folder, str):
            out_folder = RichPath.create(out_folder)
        self.__out_folder = out_folder
        self.__out_folder.make_as_dir()
        self.__file_prefix = file_prefix
        self.__max_chunk_size = max_chunk_size
        self.__file_suffix = file_suffix

        self.__num_files_written = 0  # type: int

        self.__parallel_writers = 0
        if self.__parallel_writers > 0:
            self.__writer_executors = ThreadPoolExecutor(max_workers=self.__parallel_writers)

    def __write_if_needed(self)-> None:
        if len(self.__current_chunk) < self.__max_chunk_size:
            return
        self.__flush()

    def add(self, element: T)-> None:
        self.__current_chunk.append(element)
        self.__write_if_needed()

    def add_many(self, elements: Iterable[T])-> None:
        for element in elements:
            self.add(element)

    def __flush(self)-> None:
        if len(self.__current_chunk) == 0:
            return
        outfile = self.__out_folder.join(
            '%s%03d%s' % (self.__file_prefix, self.__num_files_written, self.__file_suffix)
        )
        if self.__parallel_writers > 0:
            self.__writer_executors.submit(lambda: outfile.save_as_compressed_file(self.__current_chunk))
        else:
            outfile.save_as_compressed_file(self.__current_chunk)
        self.__current_chunk = []
        self.__num_files_written += 1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self)-> None:
        self.__flush()
        if self.__parallel_writers > 0:
            self.__writer_executors.shutdown(wait=True)
