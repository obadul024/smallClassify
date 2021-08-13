import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.applications.inception_v3 import preprocess_input


train_data_dir = 'supermen'
batch_size = 16
img_height = 299
img_width = 299
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2) # set validation split

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training') # set as training data

validation_generator = train_datagen.flow_from_directory(
    train_data_dir, # same directory as training data
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation') # set as validation data


# CREATE MODEL FROM INCEPTION_V3
# base_model loads the InceptionV3 model
# predictions add a new Dense layer with 5 neurons for our five classes
#then the model is create using base_model and predictions as its output

base_model = InceptionV3(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(5, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)



# LOAD SAVED MODEL
# When you want to run the saved model, comment the above line 46
# to load a saved model uncomment this and run it 
#model = keras.models.load_model('mymodel')
# model.summary()




# Layers are set to not trainable
# we set original InceptionV3 model to freeze, so that out Dense layer has a chance to get initialised

for layer in model.layers:
    layer.trainable = False


model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# start fitting the model to get Dense layer initialsed
model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = validation_generator,
    validation_steps = validation_generator.samples // batch_size,
    epochs = 10)


# Print layer names and then set the last layer few layers to unfreeze so we can train them to classify

for i, layer in enumerate(model.layers):
    print(i, layer.name)

for layer in model.layers[:249]:
    layer.trainable = False

for layer in model.layers[249:]:
    layer.trainable = True
    print(layer.name)
 


# Compile our Model to use Stochastic Gradient Descent Algorithm with a very low Learning Rate so it can stop Underfitting

from tensorflow.keras.optimizers import SGD
model.compile(optimizer=SGD(learning_rate=0.0001, momentum=0.9), loss='categorical_crossentropy')

model.fit_generator(
    
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = validation_generator,
    validation_steps = validation_generator.samples // batch_size,
    epochs = 30)


 #Save the model as retrained_model
model.save('retrained_model')


# TEST THIS MODEL
import glob


for filepath in glob.iglob('./test/*/*'):
    
    img = image.load_img(filepath, target_size=(img_height, img_width))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    y_prob = model.predict(x)
    print(list(train_generator.class_indices)[np.argmax(y_prob)],'--------', filepath[14:])
