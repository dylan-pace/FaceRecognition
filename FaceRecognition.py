#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt
from tensorboard.plugins.hparams import api as hp
print(tf.__version__)
#Import the necessary libararies to make tensorflow and python function.
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()
from sklearn import preprocessing, linear_model, model_selection, neighbors, svm, naive_bayes, metrics


# In[ ]:


attr_df = pd.read_csv('Downloads/faceRecognition/list_attr_celeba.csv')


# In[ ]:


attr_df = attr_df.set_index('image_id')


# In[ ]:


attr_df.replace(to_replace=-1, value=0, inplace=True)


# In[ ]:


attr_df.head()


# In[ ]:


# Female or Male?
plt.title('Female or Male')
sns.countplot(y='Male', data=attr_df, color="c")
plt.show()


# In[ ]:


part_df = pd.read_csv('Downloads/faceRecognition/list_eval_partition.csv')
part_df.head()


# In[ ]:


part_df = part_df.set_index('image_id')


# In[ ]:


# join the partition with the attributes
df_par_attr = part_df.join(attr_df['Male'], how='inner')
df_par_attr.head()


# In[ ]:


# set variables 
images_folder = 'Downloads/faceRecognition/img_align_celeba/img_align_celeba/'

EXAMPLE_PIC = images_folder + '000507.jpg'

TRAINING_SAMPLES = 10000
VALIDATION_SAMPLES = 2000
TEST_SAMPLES = 2000
IMG_WIDTH = 178
IMG_HEIGHT = 218
BATCH_SIZE = 16
NUM_EPOCHS = 20


# In[ ]:


def load_reshape_img(fname):
    img = image.load_img(fname)
    x = image.img_to_array(img)/255.
    x = x.reshape((1,) + x.shape)

    return x


def generate_df(partition, attr, num_samples):
#    partition
#        0 = train
#        1 = validation
#        2 = test
    
    df_ = df_par_attr[(df_par_attr['partition'] == partition) 
                           & (df_par_attr[attr] == 0)].sample(int(num_samples/2))
    df_ = pd.concat([df_,
                      df_par_attr[(df_par_attr['partition'] == partition) 
                                  & (df_par_attr[attr] == 1)].sample(int(num_samples/2))])

    # for Train and Validation
    if partition != 2:
        x_ = np.array([load_reshape_img(images_folder + fname) for fname in df_.index])
        x_ = x_.reshape(x_.shape[0], 218, 178, 3)
        y_ = tf.keras.utils.to_categorical(df_[attr],2)
    # for Test
    else:
        x_ = []
        y_ = []

        for index, target in df_.iterrows():
            im = cv2.imread(images_folder + index)
            im = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (IMG_WIDTH, IMG_HEIGHT)).astype(np.float32) / 255.0
            im = np.expand_dims(im, axis =0)
            x_.append(im)
            y_.append(target[attr])

    return x_, y_


# In[ ]:


# Train data
x_train, y_train = generate_df(0, 'Male', TRAINING_SAMPLES)
from keras.applications.inception_v3 import InceptionV3, preprocess_input

# Train - Data Preparation - Data Augmentation with generators
train_datagen =  tf.keras.preprocessing.image.ImageDataGenerator(
  preprocessing_function=preprocess_input,
  rotation_range=30,
  width_shift_range=0.2,
  height_shift_range=0.2,
  shear_range=0.2,
  zoom_range=0.2,
  horizontal_flip=True,
)

train_datagen.fit(x_train)

train_generator = train_datagen.flow(
x_train, y_train,
batch_size=BATCH_SIZE,
)


# In[ ]:


# Validation Data
x_valid, y_valid = generate_df(1, 'Male', VALIDATION_SAMPLES)


# In[ ]:


# Import InceptionV3 Model
inc_model = InceptionV3(include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
print("number of layers:", len(inc_model.layers))


# In[ ]:


from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
#Adding custom Layers
x = inc_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(512, activation="relu")(x)
predictions = Dense(2, activation="softmax")(x)


# In[ ]:


from keras.models import Sequential, Model
from keras.optimizers import SGD, Adam
# creating the final model 
model_ = Model(inputs=inc_model.input, outputs=predictions)

# Lock initial layers to do not be trained
for layer in model_.layers[:52]:
    layer.trainable = False

# compile the model
model_.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


hist = model_.fit_generator(train_generator
                     , validation_data = (x_valid, y_valid)
                      , steps_per_epoch= TRAINING_SAMPLES/BATCH_SIZE
                      , shuffle = True
                      , epochs= NUM_EPOCHS
                      , verbose=1
                    )

