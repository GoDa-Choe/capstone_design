import datetime
import os

a = datetime.datetime.now()
print(a.strftime("%Y%m%d_%H%M%S"))

os.mkdir("./temp/")
