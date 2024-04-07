import csv
import os

class CSVLogger:
    def __init__(self, filename, fieldnames, resume=False):
        self.filename = filename
        self.fieldnames = fieldnames
        self.resume = resume
        # Check if the file exists when resuming
        if self.resume:
            assert os.path.exists(self.filename), f"File {self.filename} does not exist. Cannot resume logging."
        # Open the file
        self._open_file()

    def _open_file(self):
        """Internal method to open the file."""
        if self.resume and os.path.exists(self.filename):
            # Append to the existing file without writing headers
            self.file = open(self.filename, 'a', newline='', encoding='utf-8')
            self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
        else:
            # Open new file or overwrite existing one and write headers
            self.file = open(self.filename, 'w', newline='', encoding='utf-8')
            self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
            self.writer.writeheader()

    def log(self, **kwargs):
        """Log metrics in the CSV file."""
        self.writer.writerow(kwargs)
        self.file.flush()  # Ensure data is written to file immediately

    def clear(self):
        """Clear the log file contents."""
        self.file.close()  # Close the current file
        self._open_file()  # Reopen the file to clear it and write headers, assuming not resuming
        print(f'{self.filename} cleared')

    def close(self):
        """Close the CSV file."""
        self.file.close()
