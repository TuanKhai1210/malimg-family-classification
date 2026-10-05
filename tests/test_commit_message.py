import unittest

from scripts.check_commit_message import validate_message


class CommitMessageTests(unittest.TestCase):
    def test_accepts_scoped_subject_and_body(self):
        self.assertEqual(
            validate_message("feat(data): add pixel-group split manifest\n\nKeep each hash in one partition."),
            [],
        )

    def test_rejects_vague_or_unscoped_subject(self):
        self.assertTrue(validate_message("update"))
        self.assertTrue(validate_message("fix: improve data loader"))
        self.assertTrue(validate_message("fix(data): adjust"))

    def test_requires_blank_line_before_body(self):
        self.assertTrue(validate_message("docs(readme): clarify project status\nExtra text"))


if __name__ == "__main__":
    unittest.main()
