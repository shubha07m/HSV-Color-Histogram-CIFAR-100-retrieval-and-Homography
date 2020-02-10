import os
import shutil
from tkinter import Tcl

m = 50000
n = 5000

src_dir = "C:/Users/shubh/Downloads/Shubh_mypy_proj/comp_vision/train/traintemp"
dst_dir = "C:/Users/shubh/Desktop/thisone"
file_list = os.listdir(src_dir)
file_list = (Tcl().call('lsort', '-dict', file_list))

for i in range(0, m, n):
    for j in range(i, i + 100):
        shutil.copy(file_list[j], dst_dir)
