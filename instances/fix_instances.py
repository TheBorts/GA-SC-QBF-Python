# Script to fix instances for the GA Solver project
# Takes out the zeros to the left of the matrix in the instance files

import sys 
import os
import numpy as np
import re


def fix_instance(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        num = 0
        num = int(lines[0].strip())  # Clean up the first line
        for i in range(0, num):
            # Remove leading zeros from the line
            lines[num + 2 + i] = lines[num + 2 + i][2 * i:]


    with open("fix_" + file_path, 'w') as f:
        f.writelines(lines)
    print(f"Fixed instance file: {file_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fix_instances.py <directory_path>")
        sys.exit(1)

    directory_path = sys.argv[1]

    if not os.path.isdir(directory_path):
        print(f"The provided path '{directory_path}' is not a valid directory.")
        sys.exit(1)

    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):  # Assuming instance files have .txt extension
            file_path = os.path.join(directory_path, filename)
            fix_instance(file_path)