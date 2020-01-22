import numpy as np
from Params import *
import ToolScripts.TimeLogger as logger
from ToolScripts.TimeLogger import log
import ToolScripts.NNLayers as NNs
from ToolScripts.NNLayers import FC, Regularize, Activate, Dropout, Bias, getParam, defineParam
from MakeData import ScipyMatMaker
import pickle
import scipy
from scipy.sparse import csr_matrix
import sys
import os
import tensorflow as tf
from tensorflow.core.protobuf import config_pb2
import math
from Sampler import Sampler, AdjLists

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Recommender:
	def __init__(self, sess, datas, inpDim):
		self.inpDim = inpDim
		self.sess = sess
		self.trnSamp, self.tstSamp = datas
		self.metrics = dict()
		self.metrics['trainLoss'] = list()
		self.metrics['trainRMSE'] = list()
		self.metrics['testLoss'] = list()
		self.metrics['testRMSE'] = list()

	def makePrint(self, name, ep, reses, save):
		ret = 'Epoch %d/%d, %s: ' % (ep, EPOCH, name)
		for metric in reses:
			val = reses[metric]
			ret += '%s = %.4f, ' % (metric, val)
			if save and name == 'Train':
				tem = 'train' + metric
				if tem in self.metrics:
					self.metrics[tem].append(val)
			elif save:
				tem = 'test' + metric
				if tem in self.metrics:
					self.metrics[tem].append(val)
		ret = ret[:-2] + '   '
		return ret

	def run(self):
		self.prepareModel()
		log('Model Prepared')
		stloc = 0
		if LOAD_MODEL != None:
			self.loadModel()
			stloc = len(self.metrics['trainLoss'])
		else:
			init = tf.global_variables_initializer()
			self.sess.run(init)
			log('Variables Inited')
		for ep in range(stloc, EPOCH):
			test = (ep % 2 == 0)
			reses = self.runEpoch(trainSamp=self.trnSamp, train=True)
			log(self.makePrint('Train', ep, reses, test))
			if test:
				final = (ep == 90)
				reses = self.runEpoch(trainSamp=self.trnSamp, labelSamp=self.tstSamp, final=final)
				log(self.makePrint('Test', ep, reses, test))
			if ep % 5 == 0:
				self.saveHistory()
			print('')
		reses =self.runEpoch(trainSamp=self.trnSamp, labelSamp=self.tstSamp)
		log(self.makePrint('Test', EPOCH, reses, True))
		self.saveHistory()

	def activateReceptors(self, data, centroids):
		receptor = lambda residual: tf.maximum(0.0, (1.0 - tf.abs(residual) /3) * 3)
		# receptor = lambda residual: 3*tf.exp(-tf.square(residual)/1)
		receptedVals = [None] * len(centroids)
		dataMask = tf.sign(data)
		for i in range(len(centroids)):
			residual = data - centroids[i]
			receptedVal = receptor(residual) * dataMask
			receptedVals[i] = receptedVal
		return receptedVals

	def multiHeadAttention(self, localReps, globalRep, number, numHeads, inpDim, name):
		attRep1 = tf.reshape(tf.tile(tf.reshape(FC(globalRep, inpDim, useBias=True, reg=True, name=name+'MHAtt_query'), [-1, 1, inpDim]), [1, number, 1]), [-1, numHeads, inpDim//numHeads])
		temLocals = tf.reshape(localReps, [-1, inpDim])
		attRep2 = tf.reshape(FC(temLocals, inpDim, useBias=True, reg=True, name=name+'MHAtt_key'), [-1, numHeads, inpDim//numHeads])
		attRep3 = tf.reshape(FC(temLocals, inpDim, useBias=True, reg=True, name=name+'MHAtt_value'), [-1, number, numHeads, inpDim//numHeads])
		# We found that the normalization introduced in the original transformer paper didn't lead to better magnitude and performance in our case
		# Instead, a constant MULT is used. By test, multiply by tf.sqrt(inpDim // numHeads) yield similar results
		att = tf.nn.softmax(tf.reshape(tf.reduce_sum(attRep1 * attRep2, axis=-1)*MULT , [-1, number, numHeads, 1]), axis=1)
		attRep = tf.reshape(tf.reduce_sum(attRep3 * att, axis=1), [-1, inpDim])
		return attRep

	def nonLinearAttention(self, localReps, globalRep, number, inpDim):
		attRep1 = tf.reshape(tf.tile(tf.reshape(FC(globalRep, ATT_DIM, reg=True, useBias=False), [-1, 1, ATT_DIM]), [1, number, 1]), [-1, ATT_DIM])
		attRep2 = FC(tf.reshape(localReps, [-1, inpDim]), ATT_DIM, reg=True, useBias=False)
		attLat = Activate(Bias(attRep1 + attRep2), 'relu')
		attScr = FC(attLat, 1, useBias=False, reg=True)
		attBias = NNs.defineParam('attBias', [1, number, 1], initializer='zeros')
		# Multiplying 32 is for magnitude adjustment. The number is same for all experiments
		if DATASET == 'ml-1m':
			att = tf.nn.softmax(tf.reshape(attScr, [-1, number, 1]) / np.sqrt(ATT_DIM) + attBias, axis=1) * 32
		else:
			att = tf.nn.softmax(tf.reshape(attScr, [-1, number, 1]) + attBias, axis=1) * 32
		attRep = tf.reduce_sum(att * localReps, axis=1)
		return attRep

	def selfAttention(self, localReps, number, inpDim, name):
		attReps = [None] * number
		for i in range(number):
			glbRep = tf.reshape(tf.slice(localReps, [0, i, 0], [-1, 1, -1]), [-1, inpDim])
			temAttRep = self.multiHeadAttention(localReps, glbRep, number=number, numHeads=ATTENTION_HEAD, inpDim=inpDim, name=name+'selfAtt') + glbRep
			fc1 = FC(temAttRep, inpDim, reg=True, useBias=True, activation='relu', name=name+'selfAtt_FC')
			unnormed = FC(fc1, inpDim, reg=True, useBias=True, name=name+'selfAtt_FC2') + temAttRep
			attReps[i] = unnormed
			# attReps[i] = self.layerNorm(unnormed, inpDim)
		attRep = tf.stack(attReps, axis=1)
		return attRep

	def transMsg(self, inps, latdim1):
		latdim2 = latdim1 // 2
		activation = 'tanh'

		recVals = self.activateReceptors(inps, self.centroids)
		divRepLst = [None] * len(self.centroids)
		for i in range(len(self.centroids)):
			divRepLst[i] = FC(recVals[i], latdim1, useBias=False, reg=True, name='recVal'+str(i))
		divReps = tf.stack(divRepLst, axis=1)

		# center-based embeddings
		actCtrReps = Activate(Bias(divReps), activation)

		# global embedding
		preLat1 = tf.reduce_sum(divReps, axis=1)
		glbRep = Activate(Bias(preLat1), activation)

		# center-based attention
		actCtrReps = (self.selfAttention(actCtrReps, number=len(self.centroids), inpDim=latdim1, name='selfAttention_'+str(i)))
		ctrAttRep = self.nonLinearAttention(actCtrReps, glbRep, number=len(self.centroids), inpDim=latdim1)

		# dense block
		# note to change number of layers when running on different datasets
		lat1 = ctrAttRep
		if DATASET == 'ml-1m':
			lat2 = lat1
		else:
			lat2 = FC(lat1, latdim1, useBias=True, activation=activation, reg=True) + lat1
		lat3 = FC(lat2, latdim1, useBias=True, activation=activation, reg=True) + lat2

		lat4 = FC(lat3, latdim2, useBias=True, reg=True, activation=activation)
		if DATASET == 'ml-1m':
			lat5 = lat4
		else:
			lat5 = FC(lat4, latdim2, useBias=True, activation=activation, reg=True) + lat4
		lat6 = FC(lat5, latdim2, useBias=True, activation=activation, reg=True) + lat5

		lat7_0 = preLat1
		lat7_1 = FC(lat6, latdim1, useBias=True, reg=True)
		lat7 = Activate((lat7_0 + lat7_1), activation)
		if DATASET == 'ml-1m':
			lat8 = lat7
		else:
			lat8 = FC(lat7, latdim1, useBias=True, activation=activation, reg=True) + lat7
		lat9 = FC(lat8, latdim1, useBias=True, activation='relu', reg=True) + lat8

		pred = FC(lat9, self.inpDim, useBias=True, reg=True, name='pred')
		return pred

	def prepareModel(self):
		self.inp = tf.placeholder(dtype=tf.float32, shape=[None, self.inpDim], name='inp')
		self.label = tf.placeholder(dtype=tf.float32, shape=[None, self.inpDim], name='label')

		self.centroids = CENTS

		pred = self.transMsg(self.inp, LAT_DIM)

		labelMask = tf.sign(self.label)
		self.squareError = tf.reduce_sum(labelMask * tf.square(pred - self.label))
		self.preLoss = tf.reduce_mean(tf.reduce_sum(labelMask * tf.square(pred - self.label), axis=-1))
		self.regLoss = REG_WEIGHT * Regularize(method='L2')
		self.loss = self.preLoss + self.regLoss

		globalStep = tf.Variable(0, trainable=False)
		learningRate = tf.train.exponential_decay(LR, globalStep, DECAY_STEP, DECAY, staircase=True)
		self.optimizer = tf.train.AdamOptimizer(learningRate).minimize(self.loss, global_step=globalStep)

	def runEpoch(self, trainSamp, labelSamp=None, train=False, final=False, steps=-1):
		inpMat = trainSamp.adjMat
		labelMat = inpMat if train else labelSamp.adjMat

		num = inpMat.shape[0]
		shuffledIds = np.random.permutation(num)
		epochLoss, epochRmse, epochNum = [0, 0, 0]
		temStep = int(np.ceil(num / BATCH_SIZE))
		if steps == -1 or steps > temStep:
			steps = temStep
		elif steps > 0 and steps < 1:
			steps = int(steps * temStep)

		for i in range(steps):
			st = i * BATCH_SIZE
			ed = min((i+1) * BATCH_SIZE, num)
			batchIds = shuffledIds[st: ed]

			sparseTrain = inpMat[batchIds]
			sparseLabel = sparseTrain if train else labelMat[batchIds]
			# A random half masking yield the same results as the random split of autoregression, and is simpler
			if CUT_ORDERING and train:
				inpMask = np.random.randint(2, size=(ed-st, self.inpDim)) * 1.0
				outMask = 1 - inpMask
				temTrain = sparseTrain.toarray() * inpMask
				temLabel = sparseLabel.toarray() * outMask
			else:
				temTrain = sparseTrain.toarray()
				temLabel = sparseLabel.toarray()

			target = [self.squareError, self.preLoss, self.loss, self.regLoss]
			if train:
				target = [self.optimizer] + target
			res = self.sess.run(target, feed_dict={self.inp: temTrain, self.label: temLabel}, options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
			sqErr, preLoss, loss, regLoss = res[-4:]

			epochLoss += loss
			epochRmse += sqErr
			epochNum += np.sum(temLabel!=0)

			log('Step %d/%d: loss = %.2f, regLoss = %.2f' %\
				(i, steps, loss, regLoss), save=False, oneline=True)

		epochRmse = np.sqrt(epochRmse / epochNum)
		epochLoss = epochLoss / steps
		ret = dict()
		ret['Loss'] = epochLoss
		ret['regLoss'] = regLoss
		ret['RMSE'] = epochRmse
		ret['PreLoss'] = preLoss
		return ret

	def saveHistory(self):
		if EPOCH == 0:
			return
		with open('History/' + SAVE_PATH + '.his', 'wb') as fs:
			pickle.dump(self.metrics, fs)

		saver = tf.train.Saver()
		saver.save(self.sess, 'Models/' + SAVE_PATH)
		log('Model Saved: %s' % SAVE_PATH)

	def loadModel(self):
		saver = tf.train.Saver()
		saver.restore(sess, 'Models/' + LOAD_MODEL)
		with open('History/' + LOAD_MODEL + '.his', 'rb') as fs:
			self.metrics = pickle.load(fs)
		log('Model Loaded')

if __name__ == '__main__':
	if len(sys.argv) != 1:
		if len(sys.argv) == 3:
			SAVE_PATH = sys.argv[1]
			LOAD_MODEL = sys.argv[2]
		else:
			SAVE_PATH = sys.argv[1]

	logger.saveDefault = True
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	log('Start')
	maker = ScipyMatMaker()
	trnMat = maker.ReadMat(TRAIN_FILE)
	tstMat = maker.ReadMat(TEST_FILE)
	log('Load Data')
	trnSamp = Sampler(trnMat)
	tstSamp = Sampler(tstMat)
	log('Sampler Inited')

	with tf.Session(config=config) as sess:
		recom = Recommender(sess, (trnSamp, tstSamp), USER_NUM if MOVIE_BASED else MOVIE_NUM)
		recom.run()