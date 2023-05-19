# Multilayer perceptron is a decoding tool 

# can introduce non-linearity

import os
import warnings
warnings.filterwarnings(action='once')


# We are going to work with the Haxby dataset [HGF+01] again. You can check the section An overview of the Haxby dataset for more details # on that dataset. Here we are going to quickly download and prepare it for machine learning applications with a set of predictive 
# variables, the brain time series X, and a dependent variable, the respective cognitive processes/function/percepts y.

from nilearn import datasets
# We are fetching the data for subject 4
data_dir = os.path.join('..', 'data')
sub_no = 4
haxby_dataset = datasets.fetch_haxby(subjects=[sub_no], fetch_stimuli=True, data_dir=data_dir)
func_file = haxby_dataset.func[0]

# mask the data
from nilearn.input_data import NiftiMasker
mask_filename = haxby_dataset.mask_vt[0]
masker = NiftiMasker(mask_img=mask_filename, standardize=True, detrend=True)
X = masker.fit_transform(func_file)

# cognitive annotations
import pandas as pd
behavioral = pd.read_csv(haxby_dataset.session_target[0], delimiter=' ')
y = behavioral['labels']

categories = y.unique()
print(categories)
print(y.shape)
print(X.shape)

# converting everything to a one-hot encoder
# in a one-hot encoder, each category is represented as a binary vector where only one element is "hot" (set to 1) and all other elements # are "cold" (set to 0). 




#Training
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)   

from keras.models import Sequential
from keras.layers import Dense

# number of unique conditions that we have
model_mlp = Sequential()
# MPL architecture 
model_mlp.add(Dense(50 , input_dim = 675, kernel_initializer="uniform", activation = 'relu')) # input
model_mlp.add(Dense(30, kernel_initializer="uniform", activation = 'relu')) # hidden layer
model_mlp.add(Dense(len(categories), activation = 'softmax')) # output layer
model_mlp.summary()

# define how MLP will learn, set loss function, optimizer, metric 
model_mlp.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# fit the model
history = model_mlp.fit(X_train, y_train, batch_size = 10,
                             epochs = 10, validation_split = 0.2)


# plot loss curve
import matplotlib.pyplot as plt
import seaborn as sns

plt.plot(history.history['loss'], color='m')
plt.plot(history.history['val_loss'], color='c')
plt.title('MLP loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc = 'upper right')

sns.despine(offset=5)

plt.show()