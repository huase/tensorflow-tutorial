def recover(p):
    import cPickle as pickle

    d = pickle.load(open(p, "rb"))
    print d
    i = 0
    for k,v in d.iteritems():
        if len(v) > 0:
            i += 1
    print i,len(d)

if __name__ == '__main__':
    recover('train-labels.p')
