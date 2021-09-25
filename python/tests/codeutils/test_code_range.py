import unittest

from dpu_utils.codeutils.text import get_code_in_range

TEST_CODE = """1234

567
 890

abcdefghijklmnop
qrs
"""

class TestCodeRange(unittest.TestCase):
    def test_get_range(self):
        self.assertEqual(get_code_in_range(
            TEST_CODE,
            (1, 1), (1, 4)
        ), "1234")

        self.assertEqual(get_code_in_range(
            TEST_CODE,
            (1, 2), (1, 2)
        ), "2")

        self.assertEqual(get_code_in_range(
            TEST_CODE,
            (1, 1), (1, 10)
        ), "1234\n")

        self.assertEqual(get_code_in_range(
            TEST_CODE,
            (1, 1), (3, 0)
        ), "1234\n\n")

        self.assertEqual(get_code_in_range(
            TEST_CODE,
            (3, 1), (4, 10)
        ), "567\n 890\n")

        self.assertEqual(get_code_in_range(
            TEST_CODE,
            (3, 2), (4, 2)
        ), "67\n 8")

        self.assertEqual(get_code_in_range(
            TEST_CODE,
            (3, 1), (6, 0)
        ), "567\n 890\n\n")

        with self.assertRaises(ValueError):
            get_code_in_range(
                TEST_CODE,
                (7, 0), (10, 0)
            )

        with self.assertRaises(ValueError):
            get_code_in_range(
                TEST_CODE,
                (10, 0), (11, 0)
            )