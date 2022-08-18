import numpy as np
from sklearn.model_selection import train_test_split

features = np.load("test_alpha.npz")['arr_0']
#labels = np.load("labels_alpha.npz")['arr_0']
print(features.shape)
print(labels.shape)

train_images, test_images, train_labels, test_labels = train_test_split(features,test_size=0.2,random_state=42)
print(train_images.shape)

np.savez_compressed("alpha_train_images.npz",train_images)
np.savez_compressed("alpha_train_labels.npz",train_labels)
np.savez_compressed("alpha_test_images.npz",test_images)
np.savez_compressed("alpha_test_labels.npz",test_labels)

