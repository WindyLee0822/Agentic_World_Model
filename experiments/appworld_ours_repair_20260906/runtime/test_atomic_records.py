import json
from pathlib import Path
import tempfile
import unittest
from atomic_records import publish


class AtomicRecordsTest(unittest.TestCase):
    def test_invalid_downloads_leave_evidence_intact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'records.jsonl'
            original = b'{"task_id":"a","success":true}\n'
            path.write_bytes(original)
            for content in [b'', b'{"task_id":', b'{"task_id":"b","success":true}\n']:
                with self.subTest(content=content):
                    with self.assertRaises(ValueError):
                        publish(path, content, 1)
                    self.assertEqual(path.read_bytes(), original)

    def test_short_snapshot_cannot_replace_missing_local_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'records.jsonl'
            with self.assertRaises(ValueError):
                publish(path, b'{"task_id":"a"}\n', 2)
            self.assertFalse(path.exists())

    def test_append_and_unchanged_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'records.jsonl'
            original = b'{"task_id":"a"}\n'
            appended = original + b'{"task_id":"b"}\n'
            path.write_bytes(original)
            self.assertEqual(publish(path, appended, 1), (2, True))
            self.assertEqual(path.read_bytes(), appended)
            self.assertEqual(publish(path, appended, 2), (2, False))
            self.assertEqual(list(Path(directory).iterdir()), [path])


if __name__ == '__main__':
    unittest.main()
