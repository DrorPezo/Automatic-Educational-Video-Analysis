import os

files = []
files_list = []
for file in os.listdir("photos"):
    if file.endswith(".jpg"):
        fpath = os.path.join("photos", file)
        fpath += ' '
        files_list.append(fpath)

files = ''.join(str(e) for e in files_list)
print(files)

script = 'python pyview.py ' + files
os.system(script)