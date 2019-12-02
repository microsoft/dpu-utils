import unittest
import pickle

from glob import iglob
import os
from tempfile import TemporaryDirectory

from dpu_utils.mlutils import BpeVocabulary


class TestBpeVocab(unittest.TestCase):
    def test(self):
        def pseudotoken_iter():
            """Create a dummy corpus by using this project."""
            for filename in iglob(os.path.join(os.path.dirname(__file__), '..', '..', "**/*.py"), recursive=True):
                with open(filename) as f:
                    yield from f.read().split('\n')

        v = BpeVocabulary(5000)
        v.create_vocabulary(pseudotoken_iter())
        text = 'for i in range(100, 2):'
        idxs = v.get_id_or_unk_for_text(text)

        self.assertEqual(v.convert_ids_to_string(idxs), text)

        with TemporaryDirectory() as tmp:
            # Test serialization
            tmp_filename = os.path.join(tmp, 'tmp.pkl')
            with open(tmp_filename, 'wb') as f:
                pickle.dump(v, f)

            with open(tmp_filename, 'rb') as f:
                v2 = pickle.load(f)

        self.assertEqual(v2.get_id_or_unk_for_text(text), idxs)
        self.assertEqual(v2.convert_ids_to_string(idxs), text)


if __name__ == '__main__':
    unittest.main()
