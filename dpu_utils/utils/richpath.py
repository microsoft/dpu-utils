import binascii
import gzip
import json
import os
import glob
import fnmatch
import zlib
import io
import pickle
import codecs
import tempfile
import re

from abc import ABC, abstractmethod
from collections import OrderedDict
from functools import total_ordering
from typing import Any, List, Optional, Iterable

from azure.storage.blob import BlockBlobService

from dpu_utils.utils.dataloading import save_json_gz, save_jsonl_gz

AZURE_PATH_PREFIX = "azure://"

__all__ = ['RichPath', 'LocalPath', 'AzurePath']

@total_ordering
class RichPath(ABC):
    def __init__(self, path: str):
        self.__path = path

    @property
    def path(self) -> str:
        return self.__path

    @staticmethod
    def create(path: str, azure_info_path: Optional[str]=None):
        if path.startswith(AZURE_PATH_PREFIX):
            # Strip off the AZURE_PATH_PREFIX:
            path = path[len(AZURE_PATH_PREFIX):]
            account_name, container_name, path = path.split('/', 2)

            with open(azure_info_path, 'r') as azure_info_file:
                azure_info = json.load(azure_info_file)
            account_info = azure_info.get(account_name)
            if account_info is None:
                raise Exception("Could not find access information for account '%s'!" % (account_name,))

            sas_token = account_info.get('sas_token')
            account_key = account_info.get('account_key')
            if sas_token is not None:
                assert not sas_token.startswith('?'), 'SAS tokens should not start with "?". Just delete it.'  #  https://github.com/Azure/azure-storage-python/issues/301
                blob_service = BlockBlobService(account_name=account_name,
                                                sas_token=sas_token)
            elif account_key is not None:
                blob_service = BlockBlobService(account_name=account_name,
                                                account_key=account_key)
            else:
                raise Exception("Access to Azure storage account '%s' requires either account_key or sas_token!" % (
                    account_name,
                ))

            # Replace environment variables in the cache location
            cache_location = account_info.get('cache_location')
            if cache_location is not None:
                def replace_by_env_var(m) -> str:
                    env_var_name = m.group(1)
                    env_var_value = os.environ.get(env_var_name)
                    if env_var_value is not None:
                        return env_var_value
                    else:
                        return env_var_name
                cache_location = re.sub('\${([^}]+)}', replace_by_env_var, cache_location)
            return AzurePath(path,
                             azure_container_name=container_name,
                             azure_blob_service=blob_service,
                             cache_location=cache_location)
        else:
            return LocalPath(path)

    def __ne__(self, other):
        return not (self == other)

    def __lt__(self, other):
        return self.path < other.path

    @abstractmethod
    def is_dir(self) -> bool:
        pass

    @abstractmethod
    def make_as_dir(self) -> None:
        pass

    @abstractmethod
    def read_as_binary(self) -> bytes:
        """Read possibly compressed binary file."""
        pass

    @abstractmethod
    def save_as_compressed_file(self, data: Any) -> None:
        pass

    def read_as_text(self) -> str:
        return self.read_as_binary().decode('utf-8')

    def read_as_json(self) -> Any:
        return json.loads(self.read_as_text(), object_pairs_hook=OrderedDict)

    def read_as_jsonl(self) -> Iterable[Any]:
        for line in self.read_as_text().splitlines():
            yield json.loads(line, object_pairs_hook=OrderedDict)

    @abstractmethod
    def read_as_pickle(self) -> Any:
        pass

    def read_by_file_suffix(self) -> Any:
        if self.path.endswith('.json.gz') or self.path.endswith('.json'):
            return self.read_as_json()
        elif self.path.endswith('.jsonl.gz') or self.path.endswith('.jsonl'):
            return self.read_as_jsonl()
        if self.path.endswith('.pkl.gz') or self.path.endswith('.pkl'):
            return self.read_as_pickle()
        raise ValueError('File suffix must be .json, .json.gz, .pkl or .pkl.gz: %s' % self.path)

    def get_filtered_files_in_dir(self, file_pattern: str) -> List['RichPath']:
        return list(self.iterate_filtered_files_in_dir(file_pattern))

    @abstractmethod
    def iterate_filtered_files_in_dir(self, file_pattern: str) -> Iterable['RichPath']:
        pass

    @abstractmethod
    def join(self, filename: str) -> 'RichPath':
        pass

    @abstractmethod
    def basename(self) -> str:
        pass

    @abstractmethod
    def get_size(self) -> int:
        pass

    @abstractmethod
    def exists(self) -> bool:
        pass


class LocalPath(RichPath):
    def __init__(self, path: str):
        super().__init__(path)

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.path == other.path

    def __hash__(self):
        return hash(self.path)

    def __repr__(self):
        return self.path

    def is_dir(self) -> bool:
        return os.path.isdir(self.path)

    def make_as_dir(self):
        os.makedirs(self.path, exist_ok=True)

    def read_as_binary(self) -> bytes:
        if self.__is_gzipped(self.path):
            with gzip.open(self.path) as f:
                return f.read()
        else:
            with open(self.path, 'rb') as f:
                return f.read()

    def read_as_pickle(self) -> Any:
        if self.__is_gzipped(self.path):
            with gzip.open(self.path) as f:
                return pickle.load(f)
        else:
            with open(self.path, 'rb') as f:
                return pickle.load(f)

    @staticmethod
    def __is_gzipped(filename: str) -> bool:
        with open(filename, 'rb') as f:
            return binascii.hexlify(f.read(2)) == b'1f8b'

    def save_as_compressed_file(self, data: Any) -> None:
        if self.path.endswith('.json.gz'):
            save_json_gz(data, self.path)
        elif self.path.endswith('.jsonl.gz'):
            save_jsonl_gz(data, self.path)
        elif self.path.endswith('.pkl.gz'):
            with gzip.GzipFile(self.path, 'wb') as outfile:
                pickle.dump(data, outfile)
        else:
            raise ValueError('File suffix must be .json.gz or .pkl.gz: %s' % self.path)

    def iterate_filtered_files_in_dir(self, file_pattern: str) -> Iterable['LocalPath']:
        yield from (LocalPath(path)
                    for path in glob.iglob(os.path.join(self.path, file_pattern)))

    def join(self, filename: str) -> 'LocalPath':
        return LocalPath(os.path.join(self.path, filename))

    def basename(self) -> str:
        return os.path.basename(self.path)

    def get_size(self) -> int:
        return os.stat(self.path).st_size

    def exists(self) -> bool:
        return os.path.exists(self.path)


class AzurePath(RichPath):
    def __init__(self, path: str, azure_container_name: str, azure_blob_service: BlockBlobService,
                 cache_location: Optional[str]):
        super().__init__(path)
        self.__container_name = azure_container_name
        self.__blob_service = azure_blob_service
        self.__cache_location = cache_location

    def __eq__(self, other):
        return (self.__class__ == other.__class__
                and self.path == other.path
                and self.__container_name == other.__container_name
                and self.__blob_service == other.__blob_service)

    def __hash__(self):
        return hash(self.path)

    def __repr__(self):
        return "%s%s/%s/%s" % (AZURE_PATH_PREFIX, self.__blob_service.account_name, self.__container_name, self.path)

    def is_dir(self) -> bool:
        blob_list = self.__blob_service.list_blobs(self.__container_name, self.path, num_results=1)
        try:
            next(iter(blob_list))
            return True
        except StopIteration:
            return False

    def make_as_dir(self) -> None:
        # Note: Directories don't really exist in blob storage.
        # Instead filenames may contain / -- thus, we have nothing to do here
        pass

    def read_as_binary(self) -> bytes:
        if self.__cache_location is None:
            return self.__read_as_binary()

        cached_file_path = self.__cache_file_locally()
        return cached_file_path.read_as_binary()

    def __cache_file_locally(self, num_retries: int=1) -> LocalPath:
        cached_file_path = os.path.join(self.__cache_location, self.__container_name, self.path)
        if not os.path.isfile(cached_file_path):
            try:
                os.makedirs(os.path.dirname(cached_file_path), exist_ok=True)
                self.__blob_service.get_blob_to_path(self.__container_name, self.path, cached_file_path)
            except Exception as e:
                if os.path.exists(cached_file_path):
                    os.remove(cached_file_path)   # On failure, remove the cached file, if it exits.
                if num_retries == 0:
                    raise e
                else:
                    self.__cache_file_locally(num_retries-1)
        return LocalPath(cached_file_path)

    def __read_as_binary(self) -> bytes:
        with io.BytesIO() as stream:
            self.__blob_service.get_blob_to_stream(self.__container_name, self.path, stream)
            stream.seek(0)
            if binascii.hexlify(stream.read(2)) != b'1f8b':
                stream.seek(0)
                return stream.read()
            stream.seek(0)
            decompressor = zlib.decompressobj(32 + zlib.MAX_WBITS)
            decompressed_data = decompressor.decompress(stream.read())
            return decompressed_data

    def read_as_pickle(self) -> Any:
        if self.__cache_location is None:
            return pickle.loads(self.read_as_binary())

        # This makes sure that we do not use a memory stream to store the temporary data:
        cached_file_path = self.__cache_file_locally()
        # We sometimes have a corrupted cache (if the process was killed while writing)
        try:
            data = cached_file_path.read_as_pickle()
        except EOFError:
            print("I: File '%s' corrupted in cache. Deleting and trying once more." % (self,))
            os.unlink(cached_file_path.path)
            cached_file_path = self.__cache_file_locally()
            data = cached_file_path.read_as_pickle()
        return data

    def save_as_compressed_file(self, data: Any):
        # TODO: Python does not have a built-in "compress stream" functionality in its standard lib
        # Thus, we just write out to a file and upload, but of course, this should be better...
        if self.path.endswith('.json.gz'):
            f = tempfile.NamedTemporaryFile(suffix='.json.gz', delete=False)
        elif self.path.endswith('.jsonl.gz'):
            f = tempfile.NamedTemporaryFile(suffix='.jsonl.gz', delete=False)
        elif self.path.endswith('.pkl.gz'):
            f = tempfile.NamedTemporaryFile(suffix='.pkl.gz', delete=False)
        else:
            raise ValueError('File suffix must be .json.gz, .jsonl.gz or .pkl.gz: %s' % self.path)
        local_temp_file = LocalPath(f.name)
        f.close()
        local_temp_file.save_as_compressed_file(data)
        self.__blob_service.create_blob_from_path(self.__container_name, self.path, local_temp_file.path)
        os.unlink(local_temp_file.path)

    def iterate_filtered_files_in_dir(self, file_pattern: str) -> Iterable['AzurePath']:
        full_pattern = os.path.join(self.path, file_pattern)
        yield from (AzurePath(blob.name,
                              azure_container_name=self.__container_name,
                              azure_blob_service=self.__blob_service,
                              cache_location=self.__cache_location)
                    for blob in self.__blob_service.list_blobs(self.__container_name, self.path)
                    if fnmatch.fnmatch(blob.name, full_pattern))

    def join(self, filename: str) -> 'AzurePath':
        return AzurePath(os.path.join(self.path, filename),
                         azure_container_name=self.__container_name,
                         azure_blob_service=self.__blob_service,
                         cache_location=self.__cache_location)

    def basename(self) -> str:
        return os.path.basename(self.path)

    def get_size(self) -> int:
        file_properties = self.__blob_service.get_blob_properties(self.__container_name, self.path)
        return file_properties.properties.content_length

    def exists(self) -> bool:
        return self.__blob_service.exists(self.__container_name, self.path)
