import ToolScripts.DataProcessor as proc
import ToolScripts.TimeLogger as logger
import os
import numpy as np
import gc
import pickle
from Params import *
import math
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import random

class ScipyMatMaker:
	def MakeOneMat(self, infile, outfile):
		data = list()
		rows = list()
		cols = list()
		with open(infile, 'r') as fs:
			for line in fs:
				arr = line.strip().split(DIVIDER)
				movieId = int(arr[1]) - 1
				userId = int(arr[0]) - 1
				rating = float(arr[2])
				if MOVIE_BASED:
					rows.append(movieId)
					cols.append(userId)
				else:
					rows.append(userId)
					cols.append(movieId)
				data.append(rating)
		if MOVIE_BASED:
			mat = csr_matrix((data, (rows, cols)), shape=(MOVIE_NUM, USER_NUM))
		else:
			mat = csr_matrix((data, (rows, cols)), shape=(USER_NUM, MOVIE_NUM))
		print(np.max(mat), mat.max(), np.max(data))
		with open(outfile, 'wb') as fs:
			pickle.dump(mat, fs)

	def ReadMat(self, file):
		with open(file, 'rb') as fs:
			ret = pickle.load(fs)
		return ret

	def CsrToLsts(self, csrMat):
		data = csrMat.data
		indices = csrMat.indices
		indptr = csrMat.indptr
		n = len(indptr) - 1
		locLsts = [None] * n
		valLsts = [None] * n
		for i in range(n):
			locLsts[i] = indices[indptr[i]: indptr[i+1]]
			valLsts[i] = data[indptr[i]: indptr[i+1]]
		return np.array(locLsts), np.array(valLsts), csrMat.shape

	def MergeLsts(self, locLsts, valLsts):
		n = len(locLsts)
		length = np.sum(list(map(lambda x: len(x), locLsts)))
		locs = [None] * length
		vals = [None] * length
		idx = 0
		for i in range(n):
			for j in range(len(locLsts[i])):
				locs[idx] = [i, locLsts[i][j]]
				vals[idx] = valLsts[i][j]
				idx += 1
		return np.array(locs), np.array(vals)

	def CsrToLst(self, csrMat, rowids):
		data = csrMat.data
		indices = csrMat.indices
		indptr = csrMat.indptr
		n = len(indptr) - 1
		locs = [None] * n
		vals = [None] * n
		length = 0
		for i in range(n):
			temlst = indices[indptr[i]: indptr[i+1]]
			locs[i] = list(map(lambda x: [rowids[i], x], temlst))
			vals[i] = list(data[indptr[i]: indptr[i+1]])
			length += len(temlst)
		ret1 = [None] * length
		ret2 = [None] * length
		idx = 0
		for i in range(n):
			for j in range(len(locs[i])):
				ret1[idx] = locs[i][j]
				ret2[idx] = vals[i][j]
				idx += 1
		return (np.array(ret1), np.array(ret2), np.array(csrMat.shape))

	def MakeMats(self):
		trainfile = 'Datasets/' + DATASET + '/ratings_' + str(RATE) + '_train.csv'
		testfile = 'Datasets/' + DATASET + '/ratings_' + str(RATE) + '_test.csv'
		self.MakeOneMat(trainfile, TRAIN_FILE)
		self.MakeOneMat(testfile, TEST_FILE)

class DataDivider:
	def DivideData(self):
		infile = 'Datasets/' + DATASET + '/' + RATING
		tem = 'Datasets/' + DATASET + '/ratings_' + str(RATE) + '_temtrain.csv'
		out1 = 'Datasets/' + DATASET + '/ratings_' + str(RATE) + '_train.csv'
		out2 = 'Datasets/' + DATASET + '/ratings_' + str(RATE) + '_test.csv'
		proc.SubDataSet(infile, tem, out2, RATE)
		proc.SubDataSet(tem, out1, out3, 0.95 )
		os.remove(tem)

if __name__ == "__main__":
	logger.log('Start')
	divider = DataDivider()
	divider.DivideData()
	logger.log('Data Divided')
	maker = ScipyMatMaker()
	maker.MakeMats()
	logger.log('Sparse Matrix Made')
