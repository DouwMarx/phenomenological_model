import matplotlib.pyplot as plt
import numpy as np

# This is based on the paper from Schmidt

def x_t(t):
    """
    Equation 9
    :return:
    """
    # Summation + of + xt + terms
    return np.sin(t)

class h_i(object):
    def __init__(self,zeta_i,f_ni):
        self.zeta_i = zeta_i
        self.f_ni = f_ni

    def h_i_factor_1(self,t):
        """
        First part of eq 13
        :param t:
        :return:
        """
        return np.exp(-self.zeta_i*2*np.pi*self.f_ni*t)

    def h_i_factor_2(self,t):
        """
        Second part of equation 13
        :param t:
        :return:
        """
        return np.sin(2*np.pi*np.sqrt(1-self.zeta_i)*self.f_ni*t)

    def h_i(self,t):
        return self.h_i_factor_1(t)*self.h_i_factor_2(t)

obj = h_i(0.001,1)
print(obj.h_i(np.linspace(10,100)))

plt.figure()
plt.plot(obj.h_i(np.linspace(10,100)))


# class q_dg(object):
#     def __init__(self,parameters):
#         self.this = that

