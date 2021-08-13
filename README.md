# WE ARE TRAINING ON FIVE CLASSES/ ACTORS 
1. We Are training the model to recognise the five actors who played superman in the last 2 decades
2. The folder structure is as follows
            supermen/
                /cavill
                /welling
                /hoeclin
                /reeves
                /routh

3. Each of these subdirectories acts as a label for the dataset
4. I took a single picture and augmented it to get a larger dataset
5. The augmentedModel.py file is given the name of the image, which has to be one of the subdirectory name, ending in a '.jpg' extension. This file resides in the smallClassify or the root directory
6. Then this script 'augmentedModel.py' is executed the folders are populated with augmented dataset
7. Do this for each of the actors. Change the filename, execute 'augmentedModel.py' script. Repeat.
8. Then See the JupyterNotebook for Comments


# To Install run

```bash

pip install -r requirements.txt
```
