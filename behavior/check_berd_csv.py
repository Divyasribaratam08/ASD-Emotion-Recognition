import csv
from pathlib import Path

file_path = Path("datasets/stimulus/word/10_1_1.csv")

with open(file_path, "r", newline="", encoding="utf-8-sig") as file:
    reader = csv.reader(file)

    header = next(reader)
    first_row = next(reader)
    second_row = next(reader)

print("Header columns:", len(header))
print("First row columns:", len(first_row))
print("Second row columns:", len(second_row))
print("First row frame index:", first_row[0])
print("First joint coordinates:", first_row[1:4])
print("Second row frame index:", second_row[0])
print("Second frame first joint:", second_row[1:4])