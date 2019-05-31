import pathlib
import shutil

prj_path = pathlib.Path().resolve()/'CBR'/'projects'/'adult_empty' # Keep track of folder path of model. 
print(str(prj_path))

# Create new
new_path = pathlib.Path().resolve()/'CBR'/'projects'/'adult_5'
shutil.copy(str(prj_path), str(new_path))