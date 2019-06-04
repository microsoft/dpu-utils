import unittest
import dpu_utils.codeutils.identifiersplitting as split

class TestSplitCamelCase(unittest.TestCase):
    def run_test(self, identifier, expected):
        actual = split.split_camelcase(identifier)
        self.assertEqual(expected, actual)

    def test_empty_string_returns_empty_list(self):
        self.run_test("", [])

    def test_single_word_is_not_split(self):
        self.run_test("variable", ["variable"])

    def test_two_words_are_split(self):
        self.run_test("camelCase", ["camel", "Case"])

    def test_three_words_are_split(self):
        self.run_test("camelCaseIdentifier", ["camel", "Case", "Identifier"])

    def test_upper_camelcase_is_split(self):
        self.run_test("CamelCase", ["Camel", "Case"])

    def test_abbreviations_are_split_correctly(self):
        self.run_test("HTMLParser", ["HTML", "Parser"])

    def test_digits_are_split(self):
        self.run_test("var12var3", ["var", "12", "var", "3"])

    def test_special_characters_are_split(self):
        self.run_test("@var-$var", ["@", "var", "-$", "var"])
