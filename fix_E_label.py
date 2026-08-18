import csv

input_file = "dataset.csv"
output_file = "dataset_fixed2.csv"

with open(input_file, "r", newline="") as file:
    rows = list(csv.DictReader(file))

changed = 0

for row in rows:
    if row["label"] == "EE":
        row["label"] = "E"
        changed += 1

with open(output_file, "w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=rows[0].keys()
    )

    writer.writeheader()
    writer.writerows(rows)

print("EE samples changed to E:", changed)
print("Total samples:", len(rows))
print("Created:", output_file)