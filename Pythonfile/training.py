import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
import os

num_classes = 5  # Indicates the number of expressions
img_rows, img_cols = 48,48 # Size of the images
batch_size = 8 # Indicates how many images to be trained at a time

train_data_dir = r'train/train' # Train directory path
validation_data_dir = r'validation/validation' # Validation directory path

# Training
# Generating more images from a single image
train_datagen = ImageDataGenerator(rescale = 1./255, 
                                   rotation_range = 30, 
                                   shear_range = 0.3, 
                                   zoom_range = 0.3, 
                                   width_shift_range = 0.4,  
                                   height_shift_range = 0.4,
                                   horizontal_flip = True, 
                                   vertical_flip = True)

# shear_range: Float. Shear Intensity (Shear angle in counter-clockwise direction in degrees)
# zoom_range: Float or [lower, upper]. Range for random zoom. If a float, [lower, upper] = [1-zoom_range, 1+zoom_range].

# Validation
# Validation is for cross checking the training data. 
validation_datagen = ImageDataGenerator(rescale = 1./255)

# More features to be added to the training data
train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                   color_mode = 'grayscale', 
                                                   target_size = (img_rows, img_cols),
                                                   batch_size = batch_size, 
                                                   class_mode = 'categorical', # Categorical ~ There are 5 categories like happy, sad, angry, neutral, surprise
                                                   shuffle = True) # Shuffling is for mixing all the data together so that the machine will remember all the expression otherwise it may forget the previous model on trainig

validation_generator = validation_datagen.flow_from_directory(validation_data_dir,
                                                               color_mode = 'grayscale', 
                                                               target_size = (img_rows, img_cols),
                                                               batch_size = batch_size, 
                                                               class_mode = 'categorical', # Categorical ~ There are 5 categories like happy, sad, angry, neutral, surprise
                                                               shuffle = True)


model = Sequential()

# Block1
model.add(Conv2D(32,(3,3), padding = 'same', kernel_initializer = 'he_normal', input_shape =(img_rows, img_cols,1)))
# 32 is the number of neurons [i.e., input size]
# (3,3) - Size of the layer
model.add(Activation('elu'))
model.add(BatchNormalization())

model.add(Conv2D(32,(3,3), padding = 'same', kernel_initializer = 'he_normal', input_shape =(img_rows, img_cols,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))


# Block2
model.add(Conv2D(64,(3,3), padding = 'same', kernel_initializer = 'he_normal', input_shape =(img_rows, img_cols,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())

model.add(Conv2D(64,(3,3), padding = 'same', kernel_initializer = 'he_normal', input_shape =(img_rows, img_cols,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Block3
model.add(Conv2D(128,(3,3), padding = 'same', kernel_initializer = 'he_normal', input_shape =(img_rows, img_cols,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())

model.add(Conv2D(128,(3,3), padding = 'same', kernel_initializer = 'he_normal', input_shape =(img_rows, img_cols,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Block4
model.add(Conv2D(256,(3,3), padding = 'same', kernel_initializer = 'he_normal', input_shape =(img_rows, img_cols,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())

model.add(Conv2D(256,(3,3), padding = 'same', kernel_initializer = 'he_normal', input_shape =(img_rows, img_cols,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Block5
model.add(Flatten())
model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Block6
model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Block7
model.add(Dense(num_classes, kernel_initializer='he_normal'))
model.add(Activation('softmax'))

print(model.summary())


from keras.optimizers import RMSprop, SGD, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


# ModelCheckpoint - Save only those with the best accuracy
checkpoint = ModelCheckpoint('Emotion_little_vgg.h5',
                            monitor = 'val_loss', # Monitor loss or not
                            mode = 'min',
                            save_best_only = True,  # Save only the best
                            verbose = 1)

# EarlyStopping - If model validation is not improving then we stop
earlystop = EarlyStopping(monitor = 'val_loss',
                         min_delta = 0, 
                         patience = 3, # If patience = 3 means that the model validation doesn't increase for 3 rounds we stop
                         verbose = 1, 
                         restore_best_weights = True)

# ReduceLROnPlateau - Reduce learning rate. If the model accuracy is not improving reduce the learning rate
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', 
                             factor = 0.2, 
                             patience = 3, 
                             verbose = 1,
                             min_delta = 0.0001)



callbacks = [earlystop, checkpoint, reduce_lr]

model.compile(loss = 'categorical_crossentropy', 
             optimizer = Adam(lr = 0.001), 
             metrics = ['accuracy'])
        
nb_train_samples = 24256
nb_validation_samples = 3006
epochs = 25 # Number of times want to be trained

history = model.fit_generator(train_generator, 
                             steps_per_epoch = nb_train_samples//batch_size,
                             epochs = epochs,
                             callbacks = callbacks, 
                             validation_data = validation_generator, 
                             validation_steps = nb_validation_samples//batch_size)