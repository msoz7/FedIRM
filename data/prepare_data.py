import os, shutil
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

TRAIN_PERCENTAGE = 0.8
DATASETS = ['isic2019', 'ham10000', 'rsna']


def data_split(data):
	# obtain class labels
	classes = data.columns[1:]

	# create empty dataframe to store train and test data
	train = pd.DataFrame(columns=data.columns)
	test = pd.DataFrame(columns=data.columns)

	# split the data into train and test sets
	for c in classes:
		class_data = data[data[c] == 1]
		train_, test_ = train_test_split(class_data, train_size=TRAIN_PERCENTAGE, random_state=42)
		train = pd.concat([train, train_])
		test = pd.concat([test, test_])

	return train, test


# ISIC 2019 dataset
if os.path.exists(DATASETS[0]) and not os.path.exists(os.path.join(DATASETS[0], 'images')):
	os.rename(os.path.join(DATASETS[0], 'ISIC_2019_Training_Input', 'ISIC_2019_Training_Input'), os.path.join(DATASETS[0], 'images'))

	# remove the 'ISIC_2019_Training_Input' folder and ISIC_2019_Training_Metadata.csv file
	shutil.rmtree(os.path.join(DATASETS[0], 'ISIC_2019_Training_Input'))
	os.remove(os.path.join(DATASETS[0], 'ISIC_2019_Training_Metadata.csv'))

	# read the 'ISIC_2019_Training_GroundTruth.csv' file
	isic_data = pd.read_csv(os.path.join(DATASETS[0], 'ISIC_2019_Training_GroundTruth.csv'))
	isic_data = isic_data.rename(columns={'image': 'ImageID'})
	isic_data['ImageID'] = isic_data['ImageID'] + '.jpg'
	isic_data.drop(['UNK'], axis=1, inplace=True)

	# split the data into train and test sets
	isic_train, isic_test = data_split(isic_data)

  # create train and test csv files
	isic_train.to_csv(os.path.join(DATASETS[0], 'train.csv'), index=False)
	isic_test.to_csv(os.path.join(DATASETS[0], 'test.csv'), index=False)

	# remove the 'ISIC_2019_Training_GroundTruth.csv' file
	os.remove(os.path.join(DATASETS[0], 'ISIC_2019_Training_GroundTruth.csv'))

	print('ISIC 2019 dataset is ready')
else:
	print('ISIC 2019 dataset not found or already prepared')


# HAM10000 dataset
if os.path.exists(DATASETS[1]) and not os.path.exists(os.path.join(DATASETS[1], 'images')):
	# create images folder
	os.makedirs(os.path.join(DATASETS[1], 'images'), exist_ok=True)

	# move the contents of the folder 'HAM10000_images_part_1' and 'HAM10000_images_part_2' to 'images' folder
	for folder in os.listdir(DATASETS[1]):
		if folder in ['HAM10000_images_part_1', 'HAM10000_images_part_2']:
			for file in os.listdir(os.path.join(DATASETS[1], folder)):
				shutil.move(os.path.join(DATASETS[1], folder, file), os.path.join(DATASETS[1], 'images'))

	# remove the files and folders except 'images' folder and 'HAM10000_metadata.csv' file
	for file in os.listdir(DATASETS[1]):
		if file not in ['images', 'HAM10000_metadata.csv']:
			if os.path.isfile(os.path.join(DATASETS[1], file)):
				os.remove(os.path.join(DATASETS[1], file))
			else:
				shutil.rmtree(os.path.join(DATASETS[1], file))

	# read the 'HAM10000_metadata.csv' file
	ham_data = pd.read_csv(os.path.join(DATASETS[1], 'HAM10000_metadata.csv'))

	# extract the image id and lesion type columns
	ham_data = ham_data[['image_id', 'dx']]
	ham_data = ham_data.rename(columns={'image_id': 'ImageID'})
	ham_data['ImageID'] = ham_data['ImageID'] + '.jpg'

	# perform one hot encoding on the lesion type column
	enc = OneHotEncoder(sparse_output=False, dtype=int)
	encoded_data = enc.fit_transform(ham_data[['dx']])
	
	# create a dataframe with the encoded data
	encoded_df = pd.DataFrame(encoded_data, columns=enc.categories_[0])

	# combine the image id and encoded data
	ham_data = pd.concat([ham_data['ImageID'], encoded_df], axis=1)
		       
	# split the data into train and test sets
	ham_train, ham_test = data_split(ham_data)

	# create train and test csv files
	ham_train.to_csv(os.path.join(DATASETS[1], 'train.csv'), index=False)
	ham_test.to_csv(os.path.join(DATASETS[1], 'test.csv'), index=False)

	# remove the 'HAM10000_metadata.csv' file
	os.remove(os.path.join(DATASETS[1], 'HAM10000_metadata.csv'))

	print('HAM10000 dataset is ready')
else:
	print('HAM10000 dataset not found or already prepared')


# RSNA Intracranial Hemorrhage Detection dataset
if os.path.exists(DATASETS[2]) and not os.path.exists(os.path.join(DATASETS[2], 'images')):
	# define the paths
	rsna_path = os.path.join(DATASETS[2], 'research', 'dept8', 'qdou', 'data', 'RSNA-ICH')

	# rename stage_2_train to images
	os.rename(os.path.join(rsna_path, 'organized', 'stage_2_train'), os.path.join(DATASETS[2], 'images'))

	# rename training.csv and testing.csv to train.csv and test.csv
	os.rename(os.path.join(rsna_path, 'training.csv'), os.path.join(DATASETS[2], 'train.csv'))
	os.rename(os.path.join(rsna_path, 'testing.csv'), os.path.join(DATASETS[2], 'test.csv'))

	# remove research folder
	shutil.rmtree(os.path.join(DATASETS[2], 'research'))

	print('RSNA dataset is ready')
else:
	print('RSNA dataset not found or already prepared')