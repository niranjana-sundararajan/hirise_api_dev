from keras.applications.inception_v3 import InceptionV3 
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten

from keras.preprocessing.image import img_to_array
from keras.applications.inception_v3 import preprocess_input
from sklearn.metrics.cluster import v_measure_score
from keras.applications.inception_v3 import InceptionV3 
from keras.applications.xception import Xception 

from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten

from keras.preprocessing.image import img_to_array
from keras.applications.inception_v3 import preprocess_input
from numpy import unique
from numpy import where
import torchvision.transforms as transforms

import os, sys 

import torchvision.transforms as transform
if __package__ is None or __package__ == '':
    # uses current directory visibility
    from hirise import Hirise_Image
    from preprocessing import Data_Preparation, Image_Loader
else:
    from . import Data_Preparation
# Define the current and parent directories and paths
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))

def tranfer_learning_encoding(folder_path = "/content/drive/MyDrive/Images/test-data/",transfer_model = "InceptionV3", test = False, verbose = False):
  transform= transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256)),transforms.Normalize(0.40655,0.1159), transforms.Grayscale(num_output_channels=3)])
  il = Image_Loader()
  dataset_all = il.generate_dataset(folder_path = folder_path, transform=transform)
  image_list = il.create_image_list(file_path = folder_path, transform = transform)
  dp = Data_Preparation()
  dataset_tensor = dp.get_dataset_tensor( dataset_all)
  len_full_tensor = len(dataset_tensor.reshape(-1))
  dataset_tensor = dataset_tensor.reshape(int(len_full_tensor/(256*256)),256*256)
  if transfer_model == "InceptionV3":
    base_model = InceptionV3(input_shape = (256,256,3),weights = 'imagenet', include_top = False)
  if transfer_model == "Xception":
    base_model = Xception(input_shape=(256, 256, 3), weights='imagenet', include_top=False)
  # mark loaded layers as not trainable
  for layer in base_model.layers:
    layer.trainable = False

  pretrained_model = Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)

  if verbose:
    print(pretrained_model.summary())

  feature_list = []
  for img in image_list:
    image = img_to_array(img)

    image = preprocess_input(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    features = pretrained_model.predict(image)
    feature_list.append(features)

  len_feature = len(feature_list[0].reshape(-1))
  cols = [f"Feature Var {i}" for i in range(len_feature)]
  feature_df = pd.DataFrame(columns=cols)
  for i in range(len(feature_list)):
    feature = feature_list[i].reshape(-1)
    feature_df.loc[len(feature_df)] = feature
  label_list = []
  labels_df = pd.DataFrame(columns=['label'])
  if test:
    for sample in tqdm(dataset_all.dataset):
        label = sample[1]
        label_list.append(label)
    labels_df['label'] = label_list
    return feature_df, labels_df
  else:
    return feature_df