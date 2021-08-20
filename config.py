import os
from pathlib import Path

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_PATH, 'data')
POLARITY_DIR = os.path.join(DATA_DIR, 'polarity')
LOG_DIR = os.path.join(BASE_PATH, 'logs')
TRAINED_MODELS_DIR = os.path.join(BASE_PATH, 'trained_models')
RESULTS_DIR = os.path.join(BASE_PATH, "results")


Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
Path(POLARITY_DIR).mkdir(parents=True, exist_ok=True)

Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
Path(TRAINED_MODELS_DIR).mkdir(parents=True, exist_ok=True)
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)



# fb dataset dirs
FACEBOOK_DATASET_DIR = os.path.join(POLARITY_DIR, 'fb', 'splitted')
FACEBOOK_DATASET_TRAIN = os.path.join(FACEBOOK_DATASET_DIR, 'train', 'train.csv')
FACEBOOK_DATASET_TEST = os.path.join(FACEBOOK_DATASET_DIR, 'test', 'test.csv')
FACEBOOK_DATASET_DEV = os.path.join(FACEBOOK_DATASET_DIR, 'dev', 'dev.csv')
FACEBOOK_DATASET = os.path.join(FACEBOOK_DATASET_DIR, 'dataset.csv')

# csfd dataset dirs
CSFD_DATASET_DIR = os.path.join(POLARITY_DIR, 'csfd', 'splitted')
CSFD_DATASET_TRAIN = os.path.join(CSFD_DATASET_DIR, 'train', 'train.csv')
CSFD_DATASET_TEST = os.path.join(CSFD_DATASET_DIR, 'test', 'test.csv')
CSFD_DATASET_DEV = os.path.join(CSFD_DATASET_DIR, 'dev', 'dev.csv')
CSFD_DATASET = os.path.join(CSFD_DATASET_DIR, 'dataset.csv')

# mallcz cz dataset dirs
MALL_DATASET_DIR = os.path.join(POLARITY_DIR, 'mallcz', 'splitted')
MALL_DATASET_TRAIN = os.path.join(MALL_DATASET_DIR, 'train', 'train.csv')
MALL_DATASET_TEST = os.path.join(MALL_DATASET_DIR, 'test', 'test.csv')
MALL_DATASET_DEV = os.path.join(MALL_DATASET_DIR, 'dev', 'dev.csv')
MALL_DATASET = os.path.join(MALL_DATASET_DIR, 'dataset.csv')


IMDB_DATASET_CL_DIR = os.path.join(POLARITY_DIR, 'imdb', 'splitted-cl')
IMDB_DATASET_CL_TRAIN = os.path.join(IMDB_DATASET_CL_DIR, 'train', 'train.csv')
IMDB_DATASET_CL_TEST = os.path.join(IMDB_DATASET_CL_DIR, 'test', 'test.csv')
IMDB_DATASET_CL_DEV = os.path.join(IMDB_DATASET_CL_DIR, 'dev', 'dev.csv')
IMDB_DATASET_CL = os.path.join(IMDB_DATASET_CL_DIR, 'dataset.csv')

IMDB_DATASET_DIR = os.path.join(POLARITY_DIR, 'imdb', 'splitted')
IMDB_DATASET_TRAIN = os.path.join(IMDB_DATASET_DIR, 'train', 'train.csv')
IMDB_DATASET_TEST = os.path.join(IMDB_DATASET_DIR, 'test', 'test.csv')
IMDB_DATASET_DEV = os.path.join(IMDB_DATASET_DIR, 'dev', 'dev.csv')
IMDB_DATASET = os.path.join(IMDB_DATASET_DIR, 'dataset.csv')

# Data splitting
FACEBOOK_ORIG = os.path.join(POLARITY_DIR, 'fb', 'original')
CSFD_ORIG = os.path.join(POLARITY_DIR, 'csfd', 'original')
MALLCZ_ORIG = os.path.join(POLARITY_DIR, 'mallcz', 'original')
IMDB_ORIG = os.path.join(POLARITY_DIR, 'imdb', 'original')

FACEBOOK_ORIGINAL_DIR = os.path.join(FACEBOOK_ORIG, 'facebook')
CSFD_ORIGINAL_DIR = os.path.join(CSFD_ORIG, 'csfd')
MALLCZ_ORIGINAL_DIR = os.path.join(MALLCZ_ORIG, 'mallcz')
IMDB_ORIGINAL_DIR = os.path.join(IMDB_ORIG,'aclImdb_v1', 'aclImdb')

Path(FACEBOOK_ORIG).mkdir(parents=True, exist_ok=True)
Path(CSFD_ORIG).mkdir(parents=True, exist_ok=True)
Path(MALLCZ_ORIG).mkdir(parents=True, exist_ok=True)
Path(IMDB_ORIG).mkdir(parents=True, exist_ok=True)


# logs
LOGGING_FORMAT= '%(asctime)s: %(levelname)s: %(name)s %(message)s'
LOGGING_DATE_FORMAT = '%m/%d/%Y %H:%M:%S'

# Important to reproduce results
RANDOM_SEED = 666

CSFD_TEST_SIZE = 0.2
# portion that will be taken from train size
CSFD_DEV_SIZE = 0.1

FACEBOOK_TEST_SIZE = 0.3
# portion that will be taken from train size
FACEBOOK_DEV_SIZE = 0.1

# Split sizes
MALLCZ_TEST_SIZE = 0.2
# portion that will be taken from train size
MALLCZ_DEV_SIZE = 0.1

IMDB_CL_TEST_SIZE = 0.1

IMDB_CL_DEV_SIZE = 0.0

# corresponds to dir names in POLARITY_DIR
DATASET_NAMES = ['fb', 'csfd', 'mallcz', 'imdb-csfd', 'csfd-imdb']