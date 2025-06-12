# Import necessary libraries
from astropy.table import Table
import numpy as np
import argparse
import os
import pandas as pd

# Parse command-line arguments
parser = argparse.ArgumentParser(description="""Make a table for cross-matching with LAMOST survey data""")
parser.add_argument("source", type=str, default="input_catalog", help="Prefix of the input catalog file (default: 'input_catalog')")
parser.add_argument("--datadir", type=str, default="", help="Path to the directory containing the input catalog file. If not provided, assumes the file is in the same directory as the script.")
cmd_args = parser.parse_args()

# Determine the directory containing the input file
if cmd_args.datadir:
    datadir = cmd_args.datadir
else:
    datadir = os.getcwd()

# Construct input file name
file_ = os.path.join(datadir, cmd_args.source + ".csv")

# Check if the file exists
if not os.path.exists(file_):
    raise FileNotFoundError(f"The file {file_} does not exist.")

# Read the catalog table using Pandas
df = pd.read_csv(file_)

# Rename columns if necessary
if "ALPHA" in df.columns:
    df.rename(columns={"ALPHA": "RAJ2000"}, inplace=True)
if "DELTA" in df.columns:
    df.rename(columns={"DELTA": "DEJ2000"}, inplace=True)

# Convert DataFrame to Astropy Table
tab = Table.from_pandas(df)

# Calculate number of rows in the table
n = len(tab["RAJ2000"])
print(f"Number of rows in the catalog: {n}")

# Prepare RA, DEC, and separation columns
sep = np.full(n, 2.0)  # Assuming a constant separation value for all objects
ra = tab["RAJ2000"]
dec = tab["DEJ2000"]

# Create a new table
table = Table([ra, dec, sep], names=('ra', 'dec', 'radius'), meta={'name': 'first table'})

# Save the new table
asciifile = os.path.splitext(os.path.basename(file_))[0] + "-coorLamost.dat"
table.write(asciifile, format="ascii.commented_header", delimiter=',', overwrite=True)

