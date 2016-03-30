from collections import OrderedDict

import math


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
        # print self.originaldata
        print self.encodeddata.values()
        print self.normalizeddata
        # print self.n
        # print self.m
        # print self.X
        # print self.Y

    def normalize(self):
        normalizeddata = OrderedDict()
        for attr, values in self.encodeddata.items():
            normalizedvalue = [(value-min(values))/(max(values)-min(values)) for value in values]
            normalizeddata[attr] = normalizedvalue
        return normalizeddata
        pass

    def getX(self):
        X = []
        i = 0
        values = self.normalizeddata.values()
        while i<self.n:
            xi = []
            j = 0
            while j<self.m:
                xi.append(values[j+1][i])
                j += 1
            X.append(xi)
            i += 1
        return X

    def getY(self):
        return self.normalizeddata.values()[-1]

    @staticmethod
    def logit(x=[], theta=[], b=0):
        i = 0
        innerproduct = 0.0
        while i < len(x):
            innerproduct += x[i] * theta[i]
            i += 1
        var = innerproduct + b
        return 1.0 / (1 + math.exp(-var))

    @staticmethod
    def vadd(x1=(),x2=()):
        i = 0
        vsum = []
        while i<len(x1):
            # print 'x1 x2',x1,x2
            vsum.append(x1[i]+x2[i])
            i += 1
        return vsum

    @staticmethod
    def deltatheta(X=(), Y=(), theta_t=(), b_t=0):
        i = 0
        vsum = [0 for x in X]
        while i<len(X):
            # print 'X[i]',X[i]
            # print 'theta_t[i]',theta_t[i]
            err = Y[i] - LR.logit(X[i],theta_t,b_t)
            # print 'err',err
            # print 'Y[i]', Y[i]
            # print 'X[i]', X[i]
            # print 'theta_t', theta_t
            # print 'b_t', b_t
            vsum = LR.vadd([x*err for x in X[i]], vsum)
            i += 1
        return vsum

    @staticmethod
    def deltab(X=(), Y=(), theta_t=(), b_t=0):
        i = 0
        vsum = 0
        while i<len(X):
            vsum  += Y[i] - LR.logit(X[i],theta_t,b_t)
            i += 1
        return vsum

    def BGD(self, theta_t=(), b_t=0, threshold=1E-5, ita=1E-3):
        maxvari = 5
        while maxvari > threshold:
            deltatheta = LR.deltatheta(self.X,self.Y,theta_t,b_t)
            theta_t = LR.vadd(theta_t,[ita*deltatheta_i for deltatheta_i in deltatheta])
            deltab = LR.deltab(self.X, self.Y, theta_t, b_t)
            b_t = b_t + ita * deltab
            maxvari = max([abs(deltatheta_i) for deltatheta_i in deltatheta]+[abs(deltab)])
            print 'deltatheta',deltatheta
            print 'deltab',deltab
            print 'theta_t',theta_t
            print 'b_t',b_t
            # break
        return theta_t, b_t

    def momentum(self, theta_t=(), b_t=0, threshold=1E-5, ita=1E-3):
        maxvari = 5
        vtheta_t = [0 for i in range(0,len(theta_t))]
        vb_t = 0
        while maxvari > threshold:
            deltatheta = LR.deltatheta(self.X,self.Y,theta_t,b_t)
            vtheta_t = LR.vadd([0.9*vtheta_ti for vtheta_ti in vtheta_t],[ita*deltatheta_i for deltatheta_i in deltatheta])
            theta_t = LR.vadd(theta_t,vtheta_t)
            deltab = LR.deltab(self.X, self.Y, theta_t, b_t)
            vb_t = 0.9*vb_t + ita * deltab
            b_t = b_t + vb_t
            maxvari = max([abs(vtheta_ti) for vtheta_ti in vtheta_t]+[abs(vb_t)])
            print 'vtheta_t',vtheta_t
            print 'vb_t',vb_t
            print 'theta_t',theta_t
            print 'b_t',b_t
            # break
        return theta_t, b_t

    def fit(self):
        # self.theta, self.b = self.BGD(self.theta,self.b)
        self.theta, self.b = self.BGD(self.theta,self.b)
        print self.theta
        print self.b
        pass

    def predict(self):
        pass

    @staticmethod
    def isnumeric(item):
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
            if LR.isnumeric(item):
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
            i = i + 1
        return dataframe


if __name__ == '__main__':
    lr = LR()
    lr.fit()
    pass
