# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 11:49:30 2019

@author: wwj
"""

# Pythono3 code to rename multiple  
# files in a directory or folder 
  
# importing os module 
import os 
  
# Function to rename multiple files 
def main(): 
    i = 0
      
    for filename in os.listdir("wallpaper"): 
        print(i)
        dst ="%02d"%i + ".jpg"
        src = filename 
          
        # rename() function will 
        # rename all the files 
        os.rename(src, dst) 
        i += 1
  
# Driver Code 
if __name__ == '__main__': 
      
    # Calling main() function 
    main() 

#%%
#coding:utf-8
#import os
path = "./a/"
dirs = os.listdir(path)
print(type(dirs))
for i in range(0,10):
 oldname = path + dirs[i]
 newname = path + "%02d"%i +".txt"
 os.rename(oldname,newname)