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
batch_size = 8
img_height = 299
img_width = 299


train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2) # set validation split

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=8,
    class_mode='categorical',
    subset='training') # set as training data

validation_generator = train_datagen.flow_from_directory(
    train_data_dir, # same directory as training data
    target_size=(img_height, img_width),
    batch_size=8,
    class_mode='categorical',
    subset='validation') # set as validation data



base_model = InceptionV3(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)


predictions = Dense(5, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)


# for layer in base_model.layers:
#     layer.trainable = False


# model.compile(optimizer='rmsprop', loss='categorical_crossentropy')


# model.fit_generator(
#     train_generator,
#     steps_per_epoch = train_generator.samples // batch_size,
#     validation_data = validation_generator, 
#     validation_steps = validation_generator.samples // batch_size,
#     epochs = 20)







for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True


from tensorflow.keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = validation_generator, 
    validation_steps = validation_generator.samples // batch_size,
    epochs = 5)


unseen_img = 'images.jpeg'
img = image.load_img(unseen_img, target_size=(img_height, img_width))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

actor_name = model.predict(x)

print(actor_name)


# idx = 0
# for data, labels in dataset:
#     print(data.shape)
#     print(data.dtype)
#     print(labels.shape)
#     print(labels.dtype)
#     idx+=1
#     print("COUNTER : ", idx)
#     # print(data)
#     normaliser = Normalization(axis=-1)
#     normaliser.adapt(data)
#     normalised_data = normaliser(data)

#     print('var %.4f' % np.var(normalised_data))

#     print('mean %.4f' % np.mean(normalised_data))



#     break
    


# dataset = keras.preprocessing.image_dataset_from_directory(
#     './supermen', 
#     batch_size=64,
#     image_size=(200, 200),
#     validation_split=0.2,
#     labels='inferred',
#     shuffle=True,

# )

