import re
from os import listdir
from concurrent.futures import ThreadPoolExecutor
from typing import List, Iterable, TypeVar, Generic, Union

from dpu_utils.utils import RichPath

T = TypeVar('T')

__all__ = ['ChunkWriter']


class ChunkWriter(Generic[T]):
    """Encapsulates writing output into chunks (multiple consecutive files).

    By setting the file_suffix to either .pkl.gz, .json.gz or .jsonl.gz
    the appropriate format will be used for the chunks.

    ChunkWriter can be used either in a context manager, ie
     ```
       with ChunkWriter(...) as writer:
           writer.add(...)
     ```

    or by explicitly invoking `close()`, ie
    ```
       writer = ChunkWriter(...)
       # Code that uses add() or add_many()
       writer.close()
    ```
    """
    def __init__(self, out_folder: Union[RichPath, str], file_prefix: str, max_chunk_size: int, file_suffix: str,
                 parallel_writers: int = 0, mode: str = 'w'):
        self.__current_chunk = []  # type: List[T]
        if isinstance(out_folder, str):
            self.__out_folder = RichPath.create(out_folder)  # type: RichPath
        else:
            self.__out_folder = out_folder  # type: RichPath
        self.__out_folder.make_as_dir()
        self.__file_prefix = file_prefix
        self.__max_chunk_size = max_chunk_size
        self.__file_suffix = file_suffix

        self.__mode = mode.lower()
        assert self.__mode in ('a', 'w'), 'Mode must be either append (a) or write (w). Given: {0}'.format(mode)

        if self.__mode == 'w':
            self.__num_files_written = 0  # 'w' mode will begin writing from scratch
        else:
            self.__num_files_written = self.__get_max_existing_index() + 1  # 'a' mode starts after the last-written file

        self.__parallel_writers = parallel_writers
        if self.__parallel_writers > 0:
            self.__writer_executors = ThreadPoolExecutor(max_workers=self.__parallel_writers)

    def __write_if_needed(self) -> None:
        if len(self.__current_chunk) < self.__max_chunk_size:
            return
        self.__flush()

    def add(self, element: T) -> None:
        self.__current_chunk.append(element)
        self.__write_if_needed()

    def add_many(self, elements: Iterable[T]) -> None:
        for element in elements:
            self.add(element)

    def __flush(self) -> None:
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

    def __enter__(self) -> 'ChunkWriter':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        self.__flush()
        if self.__parallel_writers > 0:
            self.__writer_executors.shutdown(wait=True)

    def __get_max_existing_index(self) -> int:
        """
        Returns the largest file index within the current output folder.
        """
        file_pattern = '{0}*{1}'.format(self.__file_prefix, self.__file_suffix)
        file_regex = re.compile('.*{0}([0-9]+){1}'.format(self.__file_prefix, self.__file_suffix))

        max_index = 0
        for path in self.__out_folder.iterate_filtered_files_in_dir(file_pattern):
            match = file_regex.match(path.path)
            if match is None:
                continue
            file_index = int(match.group(1))
            max_index = max(file_index, max_index)
        return max_index
