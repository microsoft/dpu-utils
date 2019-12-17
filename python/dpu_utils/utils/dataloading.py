import gzip
import json
import codecs
from collections import OrderedDict
from typing import Any, Iterator, Iterable

__all__ = ['load_json_gz', 'save_json_gz', 'load_jsonl_gz', 'save_jsonl_gz']


def load_json_gz(filename: str) -> Any:
    reader = codecs.getreader('utf-8')
    with gzip.open(filename) as f:
        return json.load(reader(f), object_pairs_hook=OrderedDict)


def save_json_gz(data: Any, filename: str) -> None:
    writer = codecs.getwriter('utf-8')
    with gzip.GzipFile(filename, 'wb') as outfile:
        json.dump(data, writer(outfile))


def load_jsonl_gz(filename: str) -> Iterator[Any]:
    """
    Iterate through gzipped JSONL files. See http://jsonlines.org/ for more.
    """
    reader = codecs.getreader('utf-8')
    with gzip.open(filename) as f:
        for line in reader(f):
            yield json.loads(line, object_pairs_hook=OrderedDict)


def save_jsonl_gz(data: Iterable[Any], filename: str) -> None:
    with gzip.GzipFile(filename, 'wb') as out_file:
        writer = codecs.getwriter('utf-8')
        for element in data:
            writer(out_file).write(json.dumps(element))
            writer(out_file).write('\n')
