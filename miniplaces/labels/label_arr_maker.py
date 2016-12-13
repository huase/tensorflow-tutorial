import pickle
import numpy as np



label_map = pickle.load(open('labels/labelMap.p'))
train_labels = pickle.load(open('labels/train-labels.p'))
val_labels = pickle.load(open('labels/val-labels.p'))
test_labels = pickle.load(open('labels/test-labels.p'))

class_size = len(label_map)

def makeOneHot(arr):
	return tuple(arr)
	ret = np.zeros((1,class_size))
	for a in arr:
		ret[:,a]=1
	return ret


def labelIds(labels):
	ret = []
	for l in labels:
		try:
			ret.append(label_map[l])
		except KeyError:
			pass
	return ret


final_train = {}
final_test = {}
final_val = {}

for i,t in enumerate(train_labels):
	final_train[t] = makeOneHot(labelIds(train_labels[t]))

for i,t in enumerate(test_labels):
	final_test[t] = makeOneHot(labelIds(test_labels[t]))

for i,t in enumerate(val_labels):
	final_val[t] = makeOneHot(labelIds(val_labels[t]))



pickle.dump(final_train, open("final_train.p","wb"))
pickle.dump(final_val, open("final_val.p", "wb"))
pickle.dump(final_test, open("final_test.p","wb"))


