from Params import *
import Params as p
import numpy as np
from ToolScripts.TimeLogger import log
from scipy.sparse import csr_matrix

class AdjLists:
	def __init__(self, locLsts, valLsts):
		self.locLsts = locLsts
		self.valLsts = valLsts

class Sampler:
	def __init__(self, adjMat):
		self.adjMat = adjMat
		if DATASET == 'foursquare':
			temmask = adjMat.toarray() > THRESHOLD
			self.adjMat = csr_matrix(adjMat.multiply(1-temmask) + THRESHOLD * temmask) / THRESHOLD * 5
		# self.sliceSmallMat()
		# self.getTP()

	def getTP(self):
		data = self.adjMat.data
		indices = self.adjMat.indices
		indptr = self.adjMat.indptr

		newdata = [None] * len(data)
		row_ind = [None] * len(data)
		col_ind = [None] * len(data)
		length = 0

		n = len(indptr) - 1
		for i in range(n):
			temlocs = indices[indptr[i]: indptr[i+1]]
			temvals = data[indptr[i]: indptr[i+1]]
			for j in range(len(temlocs)):
				row_ind[length] = temlocs[j]
				col_ind[length] = i
				newdata[length] = temvals[j]
				length += 1
		if length != len(data):
			print('FUCKED IN Sampler', length, len(data))
			exit()
		self.tpAdjMat = csr_matrix((newdata, (row_ind, col_ind)), shape=[self.adjMat.shape[1], self.adjMat.shape[0]])

	def sliceSmallMat(self):
		data = self.adjMat.data
		indices = self.adjMat.indices
		indptr = self.adjMat.indptr
		newdata = [None] * len(data)
		newrowid = [None] * len(data)
		newcolid = [None] * len(data)
		length = 0
		n = len(indptr) - 1
		for i in range(USER_NUM):
			temlocs = indices[indptr[i]: indptr[i+1]]
			temvals = data[indptr[i]: indptr[i+1]]
			for j in range(len(temlocs)):
				if temlocs[j] < MOVIE_NUM:
					newrowid[length] = i
					newcolid[length] = temlocs[j]
					newdata[length] = temvals[j]
					length += 1
		newdata = newdata[:length]
		newrowid = newrowid[:length]
		newcolid = newcolid[:length]
		self.adjMat = csr_matrix((newdata, (newrowid, newcolid)), shape=[USER_NUM, MOVIE_NUM])
