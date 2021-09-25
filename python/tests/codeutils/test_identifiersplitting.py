import unittest
from typing import List

import dpu_utils.codeutils.identifiersplitting as split

class TestSplitCamelCase(unittest.TestCase):
    def run_test(self, identifier: str, expected: List[str]) -> None:
        actual = split.split_identifier_into_parts(identifier)
        self.assertEqual(expected, actual)

    def test_empty_string_returns_empty_list(self):
        self.run_test("", [""])
        self.run_test("_", ["_"])
        self.run_test("$", ["$"])

    def test_single_word_is_not_split(self):
        self.run_test("variable", ["variable"])
        self.run_test("i", ["i"])

    def test_two_words_are_split(self):
        self.run_test("camelCase", ["camel", "case"])
        self.run_test("camelCase2", ["camel", "case", "2"])
        self.run_test("camelCase23", ["camel", "case", "23"])

    def test_three_words_are_split(self):
        self.run_test("camelCaseIdentifier", ["camel", "case", "identifier"])

    def test_upper_camelcase_is_split(self):
        self.run_test("CamelCase", ["camel", "case"])
        self.run_test("CamelCaseId", ["camel", "case", "id"])
        self.run_test("CamelCaseID", ["camel", "case", "id"])

    def test_abbreviations_are_split_correctly(self):
        self.run_test("HTMLParser", ["html", "parser"])
        self.run_test("HTML25", ["html", "25"])

    def test_digits_are_split(self):
        self.run_test("var12var3", ["var", "12", "var", "3"])

    def test_special_characters_are_split(self):
        self.run_test("@var$var", ["@", "var", "$", "var"])
        self.run_test("@var", ["@", "var"])  # C# style
        self.run_test("$var", ["$", "var"])  # PHP style
        self.run_test("$var2", ["$", "var", "2"])  # PHP style
