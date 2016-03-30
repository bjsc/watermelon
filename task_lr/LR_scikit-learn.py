#

from collections import OrderedDict
from sklearn import linear_model, decomposition, datasets

class LR(object):
    def __init__(self, datafile='data_en.txt'):
        self.datafile = datafile

        self.originaldata = self.read()
        self.encodeddata = self.discrete()
        self.normalizeddata = self.normalize()
        self.n = len(self.encodeddata.values()[0])
        self.m = len(self.encodeddata) - 2
        # self.m = 2
        self.X = self.getX()
        self.Y = self.getY()
        self.theta = [0 for i in range(0, self.m)]
        self.b = 0

        self.scikit_lr = None

    def getX(self):
        X = []
        i = 0
        values = self.normalizeddata.values()
        while i < self.n:
            xi = []
            j = 0
            while j < self.m:
                xi.append(values[j + 1][i])
                j += 1
            X.append(xi)
            i += 1
        return X

    def getY(self):
        return self.normalizeddata.values()[-1]

    @staticmethod
    def is_numeric(item):
        try:
            float(item)
            return True
        except ValueError:
            return False

    @staticmethod
    def encoding(arr):
        """

        :param arr:
        :return:
        """
        encodedarr = []
        freqmapping = {}
        for item in arr:
            freqmapping[item] = freqmapping.get(item, 0) + 1
        encodedmapping = {}
        i = 0
        for key in freqmapping.keys():
            encodedmapping[key] = i
            i += 1
        for item in arr:
            if LR.is_numeric(item):
                encodedarr.append(float(item))
            else:
                encodedarr.append(encodedmapping.get(item))
        return encodedarr

    def discrete(self):
        discreteddata = OrderedDict()
        for attr, values in self.originaldata.items():
            encodedvalue = LR.encoding(values)
            discreteddata[attr] = encodedvalue
        return discreteddata

    def normalize(self):
        normalizeddata = OrderedDict()
        for attr, values in self.encodeddata.items():
            minval = float(min(values))
            maxval = float(max(values))
            normalizedvalue = [(value - minval) / (maxval - minval) for value in values]
            normalizeddata[attr] = normalizedvalue
        return normalizeddata

    def read(self):
        fp = open(self.datafile)
        lines = fp.readlines()
        fp.close()
        items = []
        dataframe = OrderedDict()
        for line in lines:
            items.append([item.strip() for item in line.split(',')])
        i = 0
        n = len(items[0])
        while i < n:
            arr = []
            j = 1
            while j < len(items):
                arr.append(items[j][i])
                j += 1
            dataframe[items[0][i]] = arr
            i += 1
        return dataframe

    def fit(self):
        self.scikit_lr = linear_model.LogisticRegression(C=1E5)
        self.scikit_lr.fit(self.X, self.Y)
        print self.scikit_lr.get_params()
        print 'weights:', self.scikit_lr.coef_
        print 'ture Y:', self.Y
        print 'result:', self.scikit_lr.predict(self.X)
        print 'probability:', self.scikit_lr.predict_proba(self.X)


    def empirical_error(self):
        Y_estimation = self.scikit_lr.predict(self.X)
        errcnt = 0
        i = 0
        while i < len(self.X):
            xi = self.X[i]
            yi = self.Y[i]
            if not yi == Y_estimation[i]:
                errcnt += 1
            i += 1
        print "errcnt", errcnt
        return errcnt


if __name__ == '__main__':
    lr = LR()
    lr.read()
    lr.fit()
    lr.empirical_error()
    print 'read is done'