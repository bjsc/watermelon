from collections import OrderedDict
from scipy.optimize.linesearch import line_search_wolfe2, line_search_wolfe1

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
        self.empiricalerror = 0
        # print self.originaldata
        print self.encodeddata.values()
        print self.normalizeddata
        # print self.n
        # print self.m
        # print self.X
        # print self.Y

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
    def inner_product(varr1=(), varr2=()):
        ip = 0.0
        n = len(varr1)
        i = 0
        while i < n:
            ip += varr1[i] * varr2[i]
            i += 1
        return ip

    @staticmethod
    def norm_Euclidean(varr=()):
        return LR.inner_product(varr, varr)

    @staticmethod
    def logit(x=(), theta=(), b=0):
        i = 0
        innerproduct = 0.0
        while i < len(x):
            innerproduct += x[i] * theta[i]
            i += 1
        var = innerproduct + b
        return 1.0 / (1 + math.exp(-var))

    @staticmethod
    def matrix_vector_multiplication(varr1=(), varr2=()):
        n = len(varr1)
        vres = []
        i = 0
        while i < n:
            vres.append(varr1[i] * varr2[i])
            i += 1
        return vres

    @staticmethod
    def vtimes(scalar=0.0, varr=()):
        return [scalar * v for v in varr]

    @staticmethod
    def vpower(varr, power=2.0):
        return [math.pow(v, power) for v in varr]

    @staticmethod
    def vsquare(varr):
        return LR.vpower(varr, 2)

    @staticmethod
    def vsub(x1=(), x2=()):
        i = 0
        vres = []
        while i < len(x1):
            vres.append(x1[i] - x2[i])
            i += 1
        return vres

    @staticmethod
    def vadd(x1=(), x2=()):
        i = 0
        vsum = []
        while i < len(x1):
            # print 'x1 x2',x1,x2
            vsum.append(x1[i] + x2[i])
            i += 1
        return vsum

    @staticmethod
    def deltatheta(X=(), Y=(), theta_t=(), b_t=0):
        i = 0
        vsum = [0 for x in X]
        while i < len(X):
            # print 'X[i]',X[i]
            # print 'theta_t[i]',theta_t[i]
            err = Y[i] - LR.logit(X[i], theta_t, b_t)
            # print 'err',err
            # print 'Y[i]', Y[i]
            # print 'X[i]', X[i]
            # print 'theta_t', theta_t
            # print 'b_t', b_t
            vsum = LR.vadd([x * err for x in X[i]], vsum)
            i += 1
        return vsum

    @staticmethod
    def deltab(X=(), Y=(), theta_t=(), b_t=0):
        i = 0
        vsum = 0
        while i < len(X):
            vsum += Y[i] - LR.logit(X[i], theta_t, b_t)
            i += 1
        return vsum

    @staticmethod
    def line_search(x_n=(), d_n=()):
        def f():
            pass

        def fprime():
            pass

        alpha_n = line_search_wolfe1(f, fprime, x_n, d_n)
        return alpha_n

    @staticmethod
    def cg_update_d(grad_n=(), beta_t=.0, d_t=(), t=0):
        d_n = [-g for g in grad_n]
        if t > 0:
            d_n = LR.vadd(d_n, LR.vtimes(beta_t, d_n))
        return d_n

    @staticmethod
    def cg_update_beta(grad_t=(), grad_n=(), d_t=(), method="PR"):
        beta_n = 0.0
        # Fletcher-Reeves
        if method == "FR":
            beta_n = LR.inner_product(grad_n, grad_n) / LR.inner_product(grad_t, grad_t)
        # Polak-Ribiere
        elif method == "PR":
            beta_n = LR.inner_product(grad_n, LR.vsub(grad_n, grad_t)) / LR.inner_product(grad_t, grad_t)
        # Hestenes-Stiefel
        elif method == "HS":
            y_t = LR.vsub(grad_n, grad_t)
            beta_n = - LR.inner_product(grad_n, y_t) / LR.inner_product(d_t, y_t)
        # Dai-Yuan
        elif method == "DY":
            y_t = LR.vsub(grad_n, grad_t)
            beta_n = - LR.inner_product(grad_n, grad_n) / LR.inner_product(d_t, y_t)
        return max(beta_n, 0)

    @staticmethod
    def cg_update(x_n=(), alpha_n=0.0, beta_n=0.0, d_t=(), grad_t=(), grad_n=(), t=0):
        n = len(x_n)
        d_n = LR.cg_update_d(grad_n, beta_n, d_t, t)
        alpha_n = LR.line_search(x_n, d_n)
        x_n = LR.vadd(x_n, LR.vtimes(alpha_n, d_n))
        beta_n = LR.cg_update_beta(grad_t, grad_n, d_t)
        return x_n, d_n, beta_n

    def CG(self, theta_t, b_t, threshold=1E-5):

        pass

    def BGD(self, theta_t=(), b_t=0, threshold=1E-5, eta=1E-3):
        maxvari = 5
        t = 1
        while maxvari > threshold:
            deltatheta = LR.deltatheta(self.X, self.Y, theta_t, b_t)
            deltab = LR.deltab(self.X, self.Y, theta_t, b_t)
            # update theta and b
            theta_t = LR.vadd(theta_t, [eta * deltatheta_i for deltatheta_i in deltatheta])
            b_t = b_t + eta * deltab
            maxvari = max([abs(deltatheta_i) for deltatheta_i in deltatheta] + [abs(deltab)])
            t += 1
            # print 'deltatheta', deltatheta
            # print 'deltab', deltab
            # print 'theta_t', theta_t
            # print 'b_t', b_t
            if t % 10000 == 0:
                print ".",
        print "\nt=", t - 1
        return theta_t, b_t

    def momentum(self, theta_t=(), b_t=0, threshold=1E-3, eta=1E-3):
        maxvari = 5
        vtheta_t = [0 for i in range(0, len(theta_t))]
        vb_t = 0
        t = 1
        while maxvari > threshold:
            deltatheta = LR.deltatheta(self.X, self.Y, theta_t, b_t)
            deltab = LR.deltab(self.X, self.Y, theta_t, b_t)
            # update theta and b
            vtheta_t = LR.vadd([0.9 * vtheta_ti for vtheta_ti in vtheta_t],
                               [eta * deltatheta_i for deltatheta_i in deltatheta])
            theta_t = LR.vadd(theta_t, vtheta_t)
            vb_t = 0.9 * vb_t + eta * deltab
            b_t = b_t + vb_t
            maxvari = max([abs(vtheta_ti) for vtheta_ti in vtheta_t] + [abs(vb_t)])
            t += 1
            # print 'vtheta_t', vtheta_t
            # print 'vb_t', vb_t
            # print 'theta_t', theta_t
            # print 'b_t', b_t
            if t % 10000 == 0:
                print ".",
        print "\nt=", t - 1
        return theta_t, b_t

    def adam(self, theta_t=(), b_t=0, beta_m=0.9, beta_v=0.999, eta=1E-3, epsilon=1E-8, threshold=1E-5, m_t=(), v_t=()):
        varlen = len(theta_t) + 1
        if len(m_t) == 0:
            m_t = [0 for i in range(0, varlen)]
        if len(v_t) == 0:
            v_t = [0 for i in range(0, varlen)]
        maxvari = 5
        t = 1
        while maxvari > threshold:
            deltatheta = LR.deltatheta(self.X, self.Y, theta_t, b_t)
            deltab = LR.deltab(self.X, self.Y, theta_t, b_t)
            # update theta and b
            deltavar = deltatheta + [deltab]
            m_mometum = LR.vtimes(beta_m, m_t)
            m_t = LR.vadd(m_mometum, LR.vtimes(1 - beta_m, deltavar))
            v_mometum = LR.vtimes(beta_v, v_t)
            v_t = LR.vadd(v_mometum, LR.vtimes(1 - beta_v, LR.vsquare(deltavar)))
            m_estimate = LR.vtimes(1.0 / (1 - math.pow(beta_m, t)), m_t)
            v_estimate = LR.vtimes(1.0 / (1 - math.pow(beta_v, t)), v_t)
            mvm = LR.matrix_vector_multiplication(
                LR.vtimes(eta, LR.vpower(LR.vadd(v_estimate, [epsilon for x in v_estimate]), -0.5)), m_estimate)
            theta_t = LR.vadd(theta_t, mvm[:-1])
            b_t = b_t + mvm[-1]
            t += 1
            maxvari = max(mvm)
            # print 'm_estimate', m_estimate
            # print 'v_estimate', v_estimate
            # print 'mvm', mvm
            # print 'theta_t', theta_t
            # print 'b_t', b_t
            # print 't', t
            if t % 10000 == 0:
                print ".",
        print "\nt=", t - 1
        return theta_t, b_t

    def fit(self, threshold=1E-2, eta=1E-1):
        theta = self.theta
        b = self.b
        # print "=================================================================="
        # print "BGD"
        # print "=================================================================="
        # self.theta, self.b = self.BGD(theta,b, threshold=threshold, eta=eta)
        # self.empiricalerror = self.empirical_error()
        # print "theta=", self.theta
        # print "b=", self.b
        # print "empiricalerror=", self.empiricalerror
        # print "=================================================================="
        print ""
        print ""
        print "=================================================================="
        print "Momentum"
        print "=================================================================="
        self.theta, self.b = self.momentum(theta, b, threshold=threshold, eta=eta)
        self.empiricalerror = self.empirical_error()
        print "theta=", self.theta
        print "b=", self.b
        print "empiricalerror=", self.empiricalerror
        print "=================================================================="
        print ""
        print ""
        print "=================================================================="
        print "Adam"
        print "=================================================================="
        self.theta, self.b = self.adam(theta, b, threshold=threshold, eta=eta)
        self.empiricalerror = self.empirical_error()
        print "theta=", self.theta
        print "b=", self.b
        print "empiricalerror=", self.empiricalerror
        print "=================================================================="

    def empirical_error(self):
        errcnt = 0
        i = 0
        while i < len(self.X):
            xi = self.X[i]
            yi = self.Y[i]
            pi = self.logit(xi, self.theta, self.b)
            print "p_i", pi, "y_i", yi, "x_i", xi
            if pi < 0.5 and yi == 1:
                errcnt += 1
            if pi > 0.5 and yi == 0:
                errcnt += 1
            i += 1
        print "errcnt", errcnt
        return errcnt

    def predict(self):
        pass

    def normalize(self):
        normalizeddata = OrderedDict()
        for attr, values in self.encodeddata.items():
            minval = float(min(values))
            maxval = float(max(values))
            normalizedvalue = [(value - minval) / (maxval - minval) for value in values]
            normalizeddata[attr] = normalizedvalue
        return normalizeddata

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


if __name__ == '__main__':
    lr = LR()
    lr.fit()
    lr.empirical_error()
