# Data Parameters
DATASET = 'ml-1m'
# DATASET = 'ml-10m'
# DATASET = 'netflix'
# DATASET = 'foursquare'
RATE = 0.9
TOP_K = 10
if DATASET == 'ml-1m':
	USER_NUM = 6040
	MOVIE_NUM = 3706
	DIVIDER = '::'
	RATING = 'ratings.dat'
	CENTS = [1, 2, 3, 4, 5]
elif DATASET == 'ml-10m':
	USER_NUM = 69878
	MOVIE_NUM = 10677
	DIVIDER = '::'
	RATING = 'ratings.dat'
	CENTS = [0.5, 2, 3.5, 5]
elif DATASET == 'netflix':
	USER_NUM = 480189
	MOVIE_NUM = 17770
	DIVIDER = ','
	RATING = 'combined_data_5_new.txt'
	CENTS = [0.5, 2, 3.5, 5]
elif DATASET == 'foursquare':
	USER_NUM = 24748
	MOVIE_NUM = 7763
	DIVIDER = '\t'
	RATING = 'checkins'
	THRESHOLD = 5
	NEG_SAMP = 4
	CENTS = [0.5, 2, 3.5, 5]
	TOP_K = 10

# Storage Parameters
LOAD_MODEL = None
TRAIN_FILE = 'Datasets/' + DATASET + '/mats/sparseMat_'+str(RATE)+'_train'
TEST_FILE = 'Datasets/' + DATASET + '/mats/sparseMat_'+str(RATE)+'_test'

# True for ml1m, false for others
MOVIE_BASED = True#True#False

# Model Parameters
EPOCH = 120
LR = 2e-3#2e-3#2e-4#1e-3
DECAY = 0.96
BATCH_SIZE = 32#32#128#512#128
if MOVIE_BASED:
	DECAY_STEP = MOVIE_NUM // BATCH_SIZE
else:
	DECAY_STEP = USER_NUM // BATCH_SIZE

SAVE_PATH = 'tem'

REG_WEIGHT = 1e-1#1e-1#1e-2#5e-3#5e-2
SIGMA = 1

CUT_ORDERING = True

LAT_DIM = 512
ATT_DIM = 64#64#128

ATTENTION_HEAD = 8
MULT =16