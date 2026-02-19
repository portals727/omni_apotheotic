import tkinter as tk
from tkinter import filedialog
import os
import subprocess
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import StringIO

# GUI or fallback for folder
dir_path = None
try:
    root = tk.Tk()
    root.withdraw()
    dir_path = filedialog.askdirectory(title='Choose Folder with 9 Pulse Bin Files')
except Exception as e:
    print(f'GUI failed: {e}. Using fallback.')
if not dir_path:
    dir_path = input('Enter full path to folder with bin files: ')
if not dir_path or not os.path.isdir(dir_path):
    print('Invalid folder. Exiting.')
    exit()

# Find .bin files
bins = [f for f in os.listdir(dir_path) if f.endswith('.bin')]
if len(bins) != 9:
    print(f'Found {len(bins)} bins, expected 9. Proceeding.')
if not bins:
    print('No bins. Exiting.')
    exit()

# Results list
results = []
hits_data = {}  # For heatmap: bin vs patterns

# r2 commands (6502 arch, layer 2: ?v/pxr/ph, /x for patterns)
r2_cmds = """
e asm.arch=6502
aaa
/x a9558d4020a93f8d4300a9018d50004c6020~+
/x a9~+  # LDA immediate
/x 8d~+  # STA absolute
/x 4c~+  # JMP
/x 55~+  # PWM burst
/x 3f~+  # Fault mask
/x 43~+  # F3 variant
?v @0x0043  # F3 mask
pxr 128 @0x0040  # PWM shadows
ph entropy @0x0  # Desync check
ph sha256 128 @0x0  # Fingerprint
q
"""

patterns = ['a955', 'a9', '8d', '4c', '55', '3f', '43']  # Kill ops + additional

for bin_file in bins:
    path = os.path.join(dir_path, bin_file)
    try:
        output = subprocess.check_output(['r2', '-qc', r2_cmds, path], text=True, stderr=subprocess.STDOUT)
        f3_mask = re.search(r'@0x0043\s*=\s*(\d+)', output).group(1) if re.search(r'@0x0043\s*=\s*(\d+)', output) else 'N/A'
        entropy = re.search(r'entropy\s*=\s*([\d.]+)', output).group(1) if re.search(r'entropy\s*=\s*([\d.]+)', output) else 'N/A'
        sha = re.search(r'sha256\s*=\s*([a-f0-9]+)', output).group(1) if re.search(r'sha256\s*=\s*([a-f0-9]+)', output) else 'N/A'
        pwm_hits = len(re.findall(r'0x55', output))
        stub_hits = len(re.findall(r'hit', output))
        mask_hits = len(re.findall(r'0x3f|0x43|67', output))
        pattern_counts = {p: len(re.findall(p, output.lower())) for p in patterns}
        hits_data[bin_file] = pattern_counts
        results.append({'bin': bin_file, 'f3_mask': f3_mask, 'entropy': entropy, 'sha256': sha, 'pwm_55_hits': pwm_hits, 'stub_hits': stub_hits, 'mask_hits': mask_hits, **pattern_counts})
    except Exception as e:
        print(f'Error on {bin_file}: {e}')
        results.append({'bin': bin_file, 'error': str(e)})

df = pd.DataFrame(results)
print(df.to_string())
df.to_csv('nexus_results.csv')

# Heatmap: Bins vs patterns
heatmap_df = pd.DataFrame.from_dict(hits_data, orient='index')
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(heatmap_df, annot=True, cmap='viridis', ax=ax)
ax.set_title('Kill Pattern Heatmap Across Pulse Bins')
plt.show()

# Additional: Hex viewer for first bin (example)
try:
    with open(os.path.join(dir_path, bins[0]), 'rb') as f:
        hex_data = f.read().hex()
        print(f'Hex view of {bins[0]} (first 256 chars): {hex_data[:256]}')
except Exception as e:
    print(f'Hex view error: {e}')