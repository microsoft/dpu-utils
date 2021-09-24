import gzip
import random
from collections import OrderedDict
from os import PathLike
from typing import Any, Iterable, Iterator, Optional, Union

import msgpack


from dpu_utils.utils import RichPath


def load_msgpack_l_gz(filename: Union[PathLike, str]) -> Iterator[Any]:
    with gzip.open(filename) as f:
        unpacker = msgpack.Unpacker(f, raw=False, object_pairs_hook=OrderedDict)
        yield from unpacker


def save_msgpack_l_gz(data: Iterable[Any], filename: Union[PathLike, str]) -> None:
    with gzip.GzipFile(filename, "wb") as out_file:
        packer = msgpack.Packer(use_bin_type=True)
        for element in data:
            out_file.write(packer.pack(element))


def load_all_msgpack_l_gz(
    path: RichPath,
    shuffle: bool = False,
    take_only_first_n_files: Optional[int] = None,
    limit_num_yielded_elements: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> Iterator:
    """
    Iterate through all the elements of all the `.msgpack.l.gz` in a given directory.

    :param path:
    :param shuffle:
    :param take_only_first_n_files:
    :param limit_num_yielded_elements:
    :param rng:
    :return:
    """
    all_files = sorted(path.iterate_filtered_files_in_dir("*.msgpack.l.gz"))
    if take_only_first_n_files is not None:
        all_files = all_files[:take_only_first_n_files]
    if shuffle and rng is None:
        random.shuffle(all_files)
    elif shuffle:
        rng.shuffle(all_files)

    sample_idx = 0
    for msgpack_file in all_files:
        try:
            for element in load_msgpack_l_gz(msgpack_file.to_local_path().path):
                if element is not None:
                    sample_idx += 1
                    yield element
                if limit_num_yielded_elements is not None and sample_idx > limit_num_yielded_elements:
                    return
        except Exception as e:
            print(f"Error loading {msgpack_file}: {e}.")


if __name__ == "__main__":
    # A json.tool-like CLI to look into msgpack.l.gz files.
    import sys
    import json

    for datapoint in load_msgpack_l_gz(sys.argv[1]):
        print(json.dumps(datapoint, indent=2))
        print("---------------------------------------")