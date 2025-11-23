import kagglehub
import shutil
import os

# Download latest version
path = kagglehub.dataset_download("hussainsheikh03/nlp-based-cyber-security-dataset")

print("Path to dataset files:", path)

# List files in the directory
files = os.listdir(path)
print("Files in dataset:", files)

# Find the csv file
csv_file = None
for f in files:
    if f.endswith('.csv'):
        csv_file = os.path.join(path, f)
        break

if csv_file:
    destination = r"d:\github\data_analysis\final\cyber_security.csv"
    shutil.copy(csv_file, destination)
    print(f"Copied {csv_file} to {destination}")
else:
    print("No CSV file found in the dataset.")
