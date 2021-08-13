# WE ARE TRAINING ON FIVE CLASSES/ ACTORS 
1. We Are training the model to recognise the five actors who played superman in the last 2 decades
2. Create the 'supermen' folder. 
The folder structure is as follows
             ---- > supermen {cavill, welling, hoeclin, reeves, routh}
             

3. Each of these subdirectories acts as a label for the dataset
4. Add an image file to the root directory.
5. Open the augmentedModel.py, and replace the FILENAME with any of these actors names.It has to be one of the subdirectory name, ending in a '.jpg' extension. 
6. This image file resides in the smallClassify or the root directory.
7. Then this script 'augmentedModel.py' is executed the folders are populated with augmented dataset
8. Do this for each of the actors. Change the filename, execute 'augmentedModel.py' script. Repeat.
9. Then See the JupyterNotebook for Comments


# To Install this project, run

```bash

pip install -r requirements.txt
```
