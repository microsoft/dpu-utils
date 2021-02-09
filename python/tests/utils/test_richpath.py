import json
import os
import unittest
from contextlib import contextmanager
from enum import Enum
from tempfile import TemporaryDirectory

from azure.core.exceptions import ResourceExistsError
from azure.storage.blob import ContainerClient

from dpu_utils.utils import RichPath
from dpu_utils.utils import save_jsonl_gz


class AuthType(Enum):
    CONNECTION_STRING = 0
    ACCOUNT_KEY = 1
    SAS_TOKEN = 2


class TestRichPath(unittest.TestCase):
    # Note! The following are the default secrets used in Azurite, not real secrets
    # See https://github.com/Azure/Azurite for more.
    AZURITE_DEVELOPMENT_CONNECTION_STRING = "DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;"

    # Valid until 2050
    AZURITE_DEVELOPMENT_QUERY_STRING = "?sv=2018-03-28&st=2021-02-05T15%3A03%3A54Z&se=2050-02-06T15%3A03%3A00Z&sr=c&sp=racwdl&sig=fQDYpycIa3D7XZFBMIp0%2BzrukJb3Lq80gGLs9CArSHg%3D"
    AZURITE_DEVELOPMENT_ACCOUNT_KEY = "Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw=="

    def _create_test_container(self):
        client: ContainerClient = ContainerClient.from_connection_string(
            self.AZURITE_DEVELOPMENT_CONNECTION_STRING,
            container_name="test1"
        )
        try:
            client.create_container()
        except ResourceExistsError:
            pass

    @contextmanager
    def _setup_test(self, auth_type: AuthType=AuthType.CONNECTION_STRING):
        self._create_test_container()
        with TemporaryDirectory() as tmp_config, TemporaryDirectory() as cache_dir:

            test_config = {
                "devstoreaccount1": {
                    "cache_location": cache_dir,
                }
            }
            if auth_type == AuthType.ACCOUNT_KEY:
                test_config["devstoreaccount1"].update({
                    "account_key": self.AZURITE_DEVELOPMENT_ACCOUNT_KEY,
                    "endpoint": "http://localhost:10000/devstoreaccount1"
                })
            elif auth_type == AuthType.CONNECTION_STRING:
                test_config["devstoreaccount1"].update({
                    "connection_string": self.AZURITE_DEVELOPMENT_CONNECTION_STRING,
                })
            elif auth_type == AuthType.SAS_TOKEN:
                test_config["devstoreaccount1"].update({
                    "sas_token": self.AZURITE_DEVELOPMENT_QUERY_STRING,
                    "endpoint": "http://localhost:10000/devstoreaccount1"
                })
            else:
                raise Exception(f"Unknown `auth_type`: {auth_type}")

            config_path = os.path.join(tmp_config, 'config.json')
            with open(config_path, 'w') as f:
                f.write(json.dumps(test_config))

            yield config_path

    def test_connection_types(self):
        for auth_type in AuthType:
            with self.subTest(f"Test {auth_type}"), self._setup_test(auth_type=auth_type) as az_info, TemporaryDirectory() as tmp_dir:
                data_f = os.path.join(tmp_dir, 'testtext.txt')
                with open(data_f, 'w') as f:
                    f.write("hello!")
                local_path = RichPath.create(data_f)

                remote_path = RichPath.create("azure://devstoreaccount1/test1/test_text.txt", az_info)
                remote_path.copy_from(local_path)
                local_path.delete()

                self.assertEqual(remote_path.read_as_text(), "hello!")
                remote_path.delete()

    def test_simple_read_write(self):
        with self._setup_test() as az_info:
            remote_path = RichPath.create("azure://devstoreaccount1/test1/remote_path.txt", az_info)
            with TemporaryDirectory() as tmp_dir:
                data_f = os.path.join(tmp_dir, 'testdata.txt')
                with open(data_f, 'w') as f:
                    f.write("hello!")
                local_path = RichPath.create(data_f)
                self.assertEqual(local_path.read_as_text(), "hello!")
                local_size = local_path.get_size()

                remote_path.copy_from(local_path)
                self.assertTrue(local_path.exists())
                local_path.delete()
                self.assertFalse(local_path.exists())
                local_path.delete()
                with self.assertRaises(Exception):
                    local_path.delete(missing_ok=False)

            self.assertEqual(remote_path.read_as_text(), "hello!")

            # Read once again (should trigger cache)
            self.assertEqual(remote_path.read_as_text(), "hello!")

            self.assertTrue(remote_path.exists())
            self.assertTrue(remote_path.is_file())
            self.assertFalse(remote_path.is_dir())
            self.assertEqual(local_size, remote_path.get_size())

            local_path = remote_path.to_local_path()
            self.assertTrue(local_path.exists())
            os.path.exists(local_path.path)
            with open(local_path.path, 'r') as f:
                self.assertEqual(f.read(), "hello!")

            # Delete file
            remote_path.delete()
            self.assertFalse(remote_path.exists())
            remote_path.delete()  # Should not raise Exception
            with self.assertRaises(FileNotFoundError):
                remote_path.delete(missing_ok=False)

            # Other random remote_path does not exist
            remote_path = RichPath.create("azure://devstoreaccount1/test1/remote_path2.txt", az_info)
            self.assertFalse(remote_path.exists())
            self.assertFalse(remote_path.is_dir())
            self.assertFalse(remote_path.is_file())

            with self.assertRaises(Exception):
                remote_path.read_as_text()

            with self.assertRaises(Exception):
                remote_path.get_size()

    def test_read_write_compressed_files(self):
        with self._setup_test() as az_info:
            random_elements = list(range(100))
            for suffix in ('.json.gz', '.jsonl.gz', '.pkl.gz'):
                with self.subTest(f'Read/write {suffix}'):
                    remote_path = RichPath.create(f"azure://devstoreaccount1/test1/compressed/data{suffix}", az_info)
                    remote_path.save_as_compressed_file(random_elements)

                    # Read once
                    read_nums = list(remote_path.read_by_file_suffix())
                    self.assertListEqual(read_nums, random_elements)

                    # Hit Cache
                    read_nums = list(remote_path.read_by_file_suffix())
                    self.assertListEqual(read_nums, random_elements)
                    self.assertTrue(remote_path.exists())
                    self.assertTrue(remote_path.is_file())

            remote_dir = RichPath.create(f"azure://devstoreaccount1/test1/compressed/", az_info)
            self.assertTrue(remote_dir.is_dir())
            self.assertFalse(remote_dir.is_file())
            self.assertTrue(remote_dir.exists())
            remote_files = list(remote_dir.iterate_filtered_files_in_dir('*.gz'))
            self.assertEqual(len(remote_files), 3)

            for suffix in ('.json.gz', '.jsonl.gz', '.pkl.gz'):
                joined_remote = remote_dir.join(f"data{suffix}")
                self.assertTrue(joined_remote.exists())
                read_nums = list(joined_remote.read_by_file_suffix())
                self.assertListEqual(read_nums, random_elements)

            for file in remote_files:
                read_nums = list(file.read_by_file_suffix())
                self.assertListEqual(read_nums, random_elements)
                file.delete()
                self.assertFalse(file.exists())

            self.assertFalse(remote_dir.exists())
            # The directory should now be empty
            remote_files = list(remote_dir.iterate_filtered_files_in_dir('*.gz'))
            self.assertEqual(len(remote_files), 0)

    def test_copy_from(self):
        with self._setup_test() as az_info, TemporaryDirectory() as tmp_dir:
            elements = [[i, i//2] for i in range(10000)]
            tmp_local_path = RichPath.create(tmp_dir).join("sample.json.gz")
            tmp_local_path.save_as_compressed_file(elements)

            remote_path1 = RichPath.create(f"azure://devstoreaccount1/test1/sample1.json.gz", az_info)
            self.assertFalse(remote_path1.exists())

            remote_path1.copy_from(tmp_local_path)
            tmp_local_path.delete()

            self.assertFalse(tmp_local_path.exists())
            self.assertTrue(remote_path1.exists())

            read_elements = remote_path1.read_by_file_suffix()
            self.assertListEqual(elements, read_elements)

            remote_path2 = RichPath.create(f"azure://devstoreaccount1/test1/sample2.json.gz", az_info)
            remote_path2.copy_from(remote_path1)
            remote_path1.delete()

            read_elements = remote_path2.read_by_file_suffix()
            self.assertListEqual(elements, read_elements)

            read_elements = remote_path2.to_local_path().read_by_file_suffix()
            self.assertListEqual(elements, read_elements)
            remote_path2.delete()

    def test_cache_correctness(self):
        with self._setup_test() as az_info:
            random_elements = list(range(100))
            remote_path = RichPath.create(f"azure://devstoreaccount1/test1/compressed/data.jsonl.gz", az_info)
            remote_path.save_as_compressed_file(random_elements)

            # Read once
            read_nums = list(remote_path.read_by_file_suffix())
            self.assertListEqual(read_nums, random_elements)

            # Hit Cache
            read_nums = list(remote_path.read_by_file_suffix())
            self.assertListEqual(read_nums, random_elements)
            self.assertTrue(remote_path.exists())
            self.assertTrue(remote_path.is_file())

            # Update file through other means, and ensure that cache is appropriately invalidated.
            new_elements = list(range(500))
            with TemporaryDirectory() as tmp:
                path = os.path.join(tmp, 'tst.jsonl.gz')
                save_jsonl_gz(new_elements, path)
                container_client = ContainerClient.from_connection_string(self.AZURITE_DEVELOPMENT_CONNECTION_STRING,
                                                                          "test1")
                blob_client = container_client.get_blob_client("compressed/data.jsonl.gz")
                with open(path, 'rb') as f:
                    blob_client.upload_blob(f, overwrite=True)

            read_nums = list(remote_path.read_by_file_suffix())
            self.assertListEqual(read_nums, new_elements)
            self.assertTrue(remote_path.exists())
            self.assertTrue(remote_path.is_file())
