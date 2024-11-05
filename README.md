# Diet-KBQA

### Data

The raw data and processed data folders should be based under the root directory of this repo. The data is stored [here](https://drive.google.com/drive/folders/1HnCfpIGgGQAjQUEY6IX1YaIPv3lCZlmh?usp=sharing). If you include more data from NHANES or outsides, the data directory in the google drive should be updated accordingly, so everyone access to all data. 

### Tips of Code Style

1. Please Make sure that running a certain pipeline (notebook or python file) won’t break other pipelines.
2. Never import everything (*) from one of our files. Always specify the names of the functions imported. As the utils files increase, it's getting hard to track down where the function imported from.  
3. Be sure to comment the code and clean the code as you go. Please consider to keep and save the deprecated code to google drive or a specific folder to maintain the cleaniness of the code repo. 

### TODO

1. Update the user tagging process.
2. Solve the food ingredients problem.
3. Find good demonstration cases. 