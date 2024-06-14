import unittest
from tabliblib.dedup_utils import find_covering_set
class TestFindCoveringSet(unittest.TestCase):

    def test_basic_functionality(self):
        # Test the basic functionality with a small set
        input_tuples = [("file1", "string1"), ("file2", "string2"), ("file3", "string3")]
        result = find_covering_set(input_tuples)
        self.assertEqual(len(result), 3)
        self.assertTrue(("file1", "string1") in result)
        self.assertTrue(("file2", "string2") in result)
        self.assertTrue(("file3", "string3") in result)

    def test_overlap_files(self):
        # Test files with overlapping strings
        input_tuples = [("file1", "string1"), ("file1", "string2"), ("file2", "string2"), ("file3", "string3")]
        result = find_covering_set(input_tuples)
        # Only two files are needed at most to cover all strings
        self.assertTrue(len(result) <= 4 and len(result) >= 2)

    def test_single_file(self):
        # Test with all strings in a single file
        input_tuples = [("file1", "string1"), ("file1", "string2"), ("file1", "string3")]
        result = find_covering_set(input_tuples)
        self.assertEqual(len(result), 3)
        self.assertTrue(all(file == "file1" for file, string in result))

    def test_empty_input(self):
        # Test with empty input
        input_tuples = []
        result = find_covering_set(input_tuples)
        self.assertEqual(len(result), 0)

    def test_all_strings_same_file(self):
        # Test case where all strings are contained in the same file
        input_tuples = [("file1", "string1"), ("file2", "string1"), ("file3", "string1")]
        result = find_covering_set(input_tuples)
        self.assertEqual(len(result), 1)

    def test_large_overlap(self):
        # Test with a large overlap between files
        input_tuples = [("file1", "string1"), ("file2", "string1"), ("file1", "string2"), ("file2", "string2"), ("file3", "string3")]
        result = find_covering_set(input_tuples)
        self.assertTrue(len(result) <= 3 and len(result) > 1)

    def test_no_overlap(self):
        # Test with no overlap between files
        input_tuples = [("file1", "string1"), ("file2", "string2"), ("file3", "string3"), ("file4", "string4")]
        result = find_covering_set(input_tuples)
        self.assertEqual(len(result), 4)

if __name__ == '__main__':
    unittest.main()
