import tempfile
import unittest
from itertools import permutations
from typing import Set, Callable

from dpu_utils.utils import ChunkWriter, RichPath


class TestChunkWriter(unittest.TestCase):

    def test_write_read_jsonl_sequential(self):
        self.__test_write_read(lambda p: ChunkWriter(p, file_prefix='test', max_chunk_size=123, file_suffix='-test.jsonl.gz'))

    def test_write_read_jsonl_parallel(self):
        self.__test_write_read(lambda p: ChunkWriter(p, file_prefix='test', max_chunk_size=123, file_suffix='-test.jsonl.gz', parallel_writers=5))

    def test_write_read_msgpack_sequential(self):
        self.__test_write_read(lambda p: ChunkWriter(p, file_prefix='test', max_chunk_size=123, file_suffix='-test.msgpack.l.gz'), suffix='.msgpack.l.gz')

    def test_write_read_msgpack_parallel(self):
        self.__test_write_read(
                lambda p: ChunkWriter(p, file_prefix='test', max_chunk_size=123, file_suffix='-test.msgpack.l.gz', parallel_writers=5),
                suffix='.msgpack.l.gz')


    def __test_write_read(self, chunk_writer_creator: Callable[[RichPath], ChunkWriter], suffix='.jsonl.gz'):
        all_chars = [chr(65+i) for i in range(26)]
        ground_elements = set(''.join(t) for t in permutations(all_chars, 3))  # 26^3 elements

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = RichPath.create(tmp)
            with chunk_writer_creator(tmp_path) as w:
                w.add_many(ground_elements)

            # Assert that all have been seen
            stored_elements = set()  # type: Set[str]
            for file in tmp_path.get_filtered_files_in_dir('test*'+suffix):
                stored_elements.update(file.read_by_file_suffix())

            self.assertSetEqual(stored_elements, ground_elements, f'Stored elements differ len(stored)={len(stored_elements)},' \
                                                       f' len(ground)={len(ground_elements)}.' \
                                                       f' Diff {ground_elements-stored_elements}.')