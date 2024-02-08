from padasip.filters.lms import FilterLMS
import numpy as np

class myFilterLMS(FilterLMS):
    def __init__(self, n, mu=0.01, w="random",wHist=False):
        #super().__init__(n, mu, w)
        self.kind = "LMS filter"
        if type(n) == int:
            self.n = n
        else:
            raise ValueError('The size of filter must be an integer')
        self.mu = self.check_float_param(mu, 0, 1000, "mu")
        if type(w) is np.ndarray:
            self.w = w
        else:
            self.init_weights(w, self.n)
        self.w_history = np.empty((0,self.n), float)
        self.wHist = wHist


    def predict(self, x):
        """
        This function calculates the new output value `y` from input array `x`.

        **Args:**

        * `x` : input vector (1 dimension array) in length of filter.

        **Returns:**

        * `y` : output value (float) calculated from input array.

        """
        y = np.dot(self.w, x)
        return y

    def adapt(self, d, x):
        """
        Adapt weights according one desired value and its input.

        **Args:**

        * `d` : desired value (float)

        * `x` : input array (1-dimensional array)
        """
        y = np.dot(self.w, x)
        e = d - y
        self.w += self.mu * e * x
        if self.wHist:
            self.w_history = np.append(self.w_history,[self.w], axis=0)
        return y
