import pickle
import ToolScripts.Plotter as plotter
import matplotlib.pyplot as plt
import numpy as np
from Params import *

colors = ['blue', 'cyan', 'red', 'green', 'black', 'magenta', 'yellow', 'pink', 'purple', 'chocolate', 'orange', 'steelblue', 'crimson', 'lightgreen', 'salmon', 'gold', 'darkred']
lines = ['-', '--', '-.', ':']


sets = ['ml1m']
names = sets

smooth = 1
startLoc = 1
Length = 70
for j in range(0, len(sets)):
	length = Length
	val = sets[j]
	name = names[j]
	with open('History/%s.his' % val, 'rb') as fs:
		res = pickle.load(fs)
	fstMet = 'Loss'
	scdMet = 'RMSE'
	print(name, np.max(res['train'+scdMet]), np.max(res['test'+scdMet]))

	# special on previous train save
	if val == 'head8_multiple4_sumagg':
		temlen = len(res['train'+scdMet])
		newRmse = list()
		newLoss = list()
		for i in range(temlen):
			if i % 2 == 0:
				newRmse.append(res['train'+scdMet][i])
				newLoss.append(res['train'+fstMet][i])
		res['train'+scdMet] = newRmse
		res['train'+fstMet] = newLoss

	temy = [None] * 4
	temlength = len(res['test'+fstMet])
	temy[0] = np.array(res['train'+fstMet][startLoc: min(length, temlength)])
	temy[1] = np.array(res['train'+scdMet][startLoc: min(length, temlength)])
	temy[2] = np.array(res['test'+fstMet][startLoc: min(length, temlength)])
	temy[3] = np.array(res['test'+scdMet][startLoc: min(length, temlength)])
	for i in range(4):
		if len(temy[i]) < length-startLoc:
			avg = np.mean(temy[i][-5:])
			temy[i] = np.array(list(temy[i]) + [avg] * (length-temlength))
	length -= startLoc
	y = [[], [], [], []]
	for i in range(int(length/smooth)):
		if i*smooth+smooth-1 >= len(temy[0]):
			break
		for k in range(4):
			temsum = 0.0
			for l in range(smooth):
				temsum += temy[k][i*smooth+l]
			y[k].append(temsum / smooth)
	y = np.array(y)
	length = y.shape[1]
	x = np.zeros((4, length))
	for i in range(4):
		x[i] = np.array(list(range(length)))
	plt.figure(1)
	plt.subplot(221)
	plt.title('LOSS FOR TRAIN')
	plt.plot(x[0], y[0], color=colors[j], label=name)
	plt.legend()
	plt.subplot(222)
	plt.title('LOSS FOR VAL')
	plt.plot(x[2], y[2], color=colors[j], label=name)
	plt.legend()
	plt.subplot(223)
	plt.title('RMSE FOR TRAIN')
	plt.plot(x[1], y[1], color=colors[j], label=name)
	plt.legend()
	plt.subplot(224)
	plt.title('RMSE FOR VAL')
	plt.plot(x[3], y[3], color=colors[j], label=name)
	plt.legend()

plt.show()
