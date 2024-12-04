import unittest
import os
import shutil
from utils import setup_logger, create_directory, save_file, load_file, delete_directory
from utils.logger import setup_logger, close_logger

class TestLogger(unittest.TestCase):
    def setUp(self):
        # Setup a temporary log file for testing
        self.log_file = "test_logs/test.log"
        self.logger = setup_logger("TestLogger", self.log_file)

    def tearDown(self):
        # Close the logger before cleaning up log file
        close_logger(self.logger)
        if os.path.exists("test_logs"):
            shutil.rmtree("test_logs")

    def test_logger_creates_log_file(self):
        self.logger.info("Testing logger.")
        self.assertTrue(os.path.exists(self.log_file), "Log file was not created.")

    def test_logger_logs_messages(self):
        self.logger.info("Testing message.")
        with open(self.log_file, "r") as file:
            log_content = file.read()
        self.assertIn("Testing message.", log_content, "Log message not found in log file.")

class TestFileManager(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_data"
        self.test_file = os.path.join(self.test_dir, "example.txt")
        self.test_content = b"Hello, World!"

    def tearDown(self):
        # Clean up test directory after testing
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_create_directory(self):
        create_directory(self.test_dir)
        self.assertTrue(os.path.exists(self.test_dir), "Directory was not created.")

    def test_save_file(self):
        create_directory(self.test_dir)
        save_file(self.test_content, self.test_file)
        self.assertTrue(os.path.exists(self.test_file), "File was not saved.")

    def test_load_file(self):
        create_directory(self.test_dir)
        save_file(self.test_content, self.test_file)
        content = load_file(self.test_file)
        self.assertEqual(content, self.test_content, "Loaded content does not match saved content.")

    def test_delete_directory(self):
        create_directory(self.test_dir)
        delete_directory(self.test_dir)
        self.assertFalse(os.path.exists(self.test_dir), "Directory was not deleted.")

if __name__ == "__main__":
    unittest.main()
