# link to SVM BrainHack tutorial: https://main-educational.github.io/brain_encoding_decoding/svm_decoding.html
# split the dataset into test and train, the actual model fitting is just two lines of code, very underwhelming 
from sklearn.svm import SVC
model_svm = SVC(random_state=0, kernel='linear', C=1)
model_svm.fit(X_train, y_train)

from sklearn.metrics import classification_report
y_train_pred = model_svm.predict(X_train)
print(classification_report(y_train, y_train_pred))

# confusion matrix
import sys
import numpy as np
from sklearn.metrics import confusion_matrix
sys.path.append('../src')
import visualization
cm_svm = confusion_matrix(y_test, y_test_pred)
model_conf_matrix = cm_svm.astype('float') / cm_svm.sum(axis=1)[:, np.newaxis]

visualization.conf_matrix(model_conf_matrix,
                          categories,


# Visualizing the weights
# Finally we can visualize the weights of the (linear) classifier to see which brain region seem to impact most the decision

