import csv

class CSVLogger:
    def __init__(self, filename, fieldnames):
        self.filename = filename
        self.fieldnames = fieldnames
        # Open the file and write headers
        self._open_file()

    def _open_file(self):
        """Internal method to open the file and write headers."""
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
        self._open_file()  # Reopen the file to clear it and write headers
        print(f'{self.filename} cleared')

    def close(self):
        """Close the CSV file."""
        self.file.close()