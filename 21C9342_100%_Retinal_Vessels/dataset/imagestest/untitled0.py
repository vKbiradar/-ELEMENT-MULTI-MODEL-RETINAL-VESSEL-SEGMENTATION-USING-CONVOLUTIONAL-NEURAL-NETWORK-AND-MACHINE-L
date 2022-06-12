# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 13:58:22 2022

@author: admin
"""

import os
  
# Folder Path
path = "E:/21C9342_retinal_vessels/DRIVE/test/images"
  
# Change the directory
os.chdir(path)
  
# Read text File
  
  
def read_text_file(file_path):
    with open(file_path, 'r') as f:
        print(f.read())
  
  
# iterate through all file
for file in os.listdir():
    # Check whether file is in text format or not
    if file.endswith(".tif"):
        file_path = f"{file}"
        #print(file_path)
        print(file_path.replace('.tif',""))
        
  
        # call read text file function
        