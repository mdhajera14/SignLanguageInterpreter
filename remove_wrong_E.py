import csv

input_file = "dataset.csv"
output_file = "dataset_fixed.csv"

# Number of incorrect E samples to remove
WRONG_SAMPLES = 41

with open(input_file, "r", newline="") as file:
    rows = list(csv.DictReader(file))

# Find the positions of all E samples
e_positions = [
    i for i, row in enumerate(rows)
    if row["label"] == "E"
]

print("E samples before:", len(e_positions))

if len(e_positions) < WRONG_SAMPLES:
    print("ERROR: There aren't 41 E samples to remove.")
    raise SystemExit

# Remove the LAST 41 E samples
positions_to_remove = set(e_positions[-WRONG_SAMPLES:])

rows_fixed = [
    row for i, row in enumerate(rows)
    if i not in positions_to_remove
]

# Save corrected dataset
with open(output_file, "w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=rows[0].keys()
    )

    writer.writeheader()
    writer.writerows(rows_fixed)

print("E samples after:", sum(row["label"] == "E" for row in rows_fixed))
print("Total samples after:", len(rows_fixed))
print()
print("Created:", output_file)