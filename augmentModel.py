import cv2
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
# load the image

FILENAME = 'cavill.jpg'
img = load_img(FILENAME)
# convert to numpy array
data = img_to_array(img)
# expand dimension to one sample
samples = expand_dims(data, 0)
# create image data augmentation generator
datagen = [ImageDataGenerator(width_shift_range=[-200,200]), ImageDataGenerator(height_shift_range=0.5), ImageDataGenerator(horizontal_flip=True), ImageDataGenerator(rotation_range=90), ImageDataGenerator(brightness_range=[0.2,1.0]) ]


for idx, gen in enumerate(datagen):
  # prepare iterator
  # print(idx, gen)
  it = gen.flow(samples, batch_size=1)
  # generate samples and plot
  for i in range(9):
    # define subplot
    pyplot.subplot(330 + 1 + i)
    # generate batch of images
    batch = it.next()
    # convert to unsigned integers for viewing
    image = batch[0].astype('uint8')
    # plot raw pixel data
    pyplot.imshow(image)
    cv2.imwrite('./supermen/'+FILENAME[:-4]+'/'+FILENAME[:-4]+str(i)+str(idx)+'.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
  
  

# pyplot.show()
# 