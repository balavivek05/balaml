# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 13:19:45 2021

@author: bala.vivek
"""

import re
import os, time
now=time.time()
import sys

sys.stdout=open("D:\EAI\EFTServer\Logs_globalscape.txt","w")
path = "path"
#delets the file based on the date modified.
for root, subFolder, files in os.walk(path):
    for item in files:
        if item.endswith(".log"):
            fileNamePath = str(os.path.join(root,item)) 
            if os.path.getmtime(fileNamePath) < now - 365 * 86400:
                print('The files which are removed are',fileNamePath)
                os.remove(fileNamePath)

#delete the file based on the filename.
path = "path"              
for root, subFolder, files in os.walk(path):
    for item in files:
        if item.endswith(".log"):
            fileNamePath = str(os.path.join(root,item)) 
            date_time=re.search(r'\d+', fileNamePath).group(0)
            pattern = '%y%d%m'
            epoch = int(time.mktime(time.strptime(date_time, pattern)))
            if epoch < now - 360 * 86400: 
                print('The files which are removed are',fileNamePath)
                os.remove(fileNamePath)

sys.stdout.close()
quit()
