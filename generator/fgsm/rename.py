from glob import glob
import sys
from os import rename

i = 0
f = glob(sys.argv[1]+'*')
for file in f:
    rename(f, 'val_'+str(i)+'.JPEG')
    i += 1

