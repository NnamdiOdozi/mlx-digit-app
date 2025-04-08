#!/usr/bin/env python
import os
import sys
import unittest
import datetime
from io import StringIO

# Adjust sys.path so the 'app' directory is available to tests.
project_root = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.join(project_root, "app")
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

# Discover tests: adjust the directory if your tests folder has a different name.
loader = unittest.TestLoader()
suite = loader.discover('tests')

# Set up an in-memory stream to capture test output.
output_stream = StringIO()
runner = unittest.TextTestRunner(stream=output_stream, verbosity=2)
result = runner.run(suite)

# Get the string output from the tests.
test_output = output_stream.getvalue()

# Get a timestamp.
timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Format the log entry.
log_entry = f"=== Test run at {timestamp} ===\n" + test_output + "\n"

# Append the log entry to the results file.
log_file = os.path.join(project_root,"tests", "test_results.log")
with open(log_file, "a") as f:
    f.write(log_entry)

# Also print the output to the console, if desired.
print(test_output)

# Exit with an appropriate exit code (0 if tests passed; 1 otherwise)
sys.exit(not result.wasSuccessful())
