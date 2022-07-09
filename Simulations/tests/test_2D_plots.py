import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import math
from scipy.stats import multivariate_normal

def create_multivar_gauss(mu, cov, dim_shape, show=False):
    x_min, x_max, y_min, y_max = dim_shape
    x, y = np.mgrid[x_min:x_max:1, y_min:y_max:1]
    pos = np.dstack((x, y))
    rv = multivariate_normal(mu, cov)
    if show:
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        print(rv.pdf(pos))
        ax2.imshow(rv.pdf(pos))
        plt.show()
    return np.array(rv)

# ----- OPTIMIZATION PARAMETERS ------
sig_x = 350
sig_y = 400
sig_xy = 350
mu_x = 12
mu_y = 12
mu = [mu_x, mu_y]
cov = np.array([[sig_x, sig_xy], [sig_xy, sig_y]])
# ------------------------------------

x_min = -5
x_max = 30
y_min = -5
y_max = 30
dim_shape = x_min, x_max, y_min, y_max

multivar_gauss_perf = create_multivar_gauss(mu, cov, dim_shape, show=True)

def diff_multivars(params, dim_shape, measurements):
    mu_x, mu_y, sig_x, sig_y, sig_xy = params
    mu = [mu_x, mu_y]
    cov = np.array([[sig_x, sig_xy], [sig_xy, sig_y]])
    temp_multivar = create_multivar_gauss(mu, cov, dim_shape)
    pass
    #return (messungen_verteilung - messungen)**2


# Eine Messung ist der Wert der entlang eine linie mit stützvektor t und richtungsvektor r integriert ist
# -> Die Werte der Messungen hab ich.
# -> Directional Multivariate Distribution integral muss ich Googlen...
# -> next step wäre dann nicht nur eine zu fitten, sondern so viele wie die Messung erlaubt
# -> eventuell eine klasse für die multivariates erstellen, sodass bei bedarf eine neue erstellt werden kann?

# -> erstelle die verteilung so, dass der unterschied zwischen der messung in der verteilung und die messung
#    möglichst klein ist

# -> wenn zwei messungen einen hohen wert aufweisen und sich nicht kreuzen, könnte man mit einer zweiten wolke rechnen!





















