import numpy as np
import matplotlib.pyplot as plt


def signed_power(value, power):
    if not power:
        return np.sign(value)
    else:
        return np.sign(value) * (np.abs(value)) ** power


def calculate_params(tot):
    if tot > 12:
        raise ValueError(r'Please choose a smaller order of $n_d + n_f$')
    else:
        params = [1.1, 1.5, 2, 3, 5, 7, 10, 12, 14, 17, 20, 26, 32]
        res = np.zeros(tot + 1, dtype=float)
        res[0] = params[0]
        res[tot] = params[tot]
        for i in range(tot - 1, 0, -1):
            res[i] = params[i] * (res[i + 1] ** (i/(i+1)))
    return res


def differentiator(signal, nd, nf, dt, l, tf=20, iters=0):
    if not iters:
        iters = int(tf/dt)
    ord = nd + nf
    params = calculate_params(nd + nf)
    w = np.zeros((nf, iters))
    z = np.zeros((nd + 1, iters))
    for i in range(iters - 1):
        for j in range(nf - 1):
            w[j, i+1] = w[j, i] + dt *\
                        (-params[ord-j] * l**((j+1)/(ord+1)) * signed_power(w[0, i], (ord-j)/(ord+1)) + w[j+1, i])
        w[nf-1, i+1] = w[nf-1, i] + dt *\
                       (-params[nd+1] * l**(nf/(ord+1)) * signed_power(w[0, i], (nd+1)/(ord+1)) + z[0, i] - signal[i])
        for j in range(nd):
            temp = z[nd, i]
            for p in range(nd - 1, j, -1):
                temp = dt/(p-j+1) * temp + z[p, i]
            z[j, i+1] = z[j, i] + dt *\
                        (-params[nd-j] * l**((nf+j+1)/(ord+1)) * signed_power(w[0, i], (nd-j)/(ord+1)) + temp)
        z[nd, i+1] = z[nd, i] + dt * (-params[0] * l * signed_power(w[0, i], 0))
    return z, w


def generalized_differentiator(signal, nd, nf, dt, l):
    diffs = [dt[i + 1] - dt[i] for i in range(len(dt) - 1)]
    iters = len(diffs) + 1
    ord = nd + nf
    params = calculate_params(nd + nf)
    w = np.zeros((nf, iters))
    z = np.zeros((nd + 1, iters))
    for i in range(iters - 1):
        for j in range(nf - 1):
            w[j, i+1] = w[j, i] + diffs[i] *\
                        (-params[ord-j] * l**((j+1)/(ord+1)) * signed_power(w[0, i], (ord-j)/(ord+1)) + w[j+1, i])
        w[nf-1, i+1] = w[nf-1, i] + diffs[i] *\
                       (-params[nd+1] * l**(nf/(ord+1)) * signed_power(w[0, i], (nd+1)/(ord+1)) + z[0, i] - signal[i])
        for j in range(nd):
            temp = z[nd, i]
            for p in range(nd - 1, j, -1):
                temp = diffs[i]/(p-j+1) * temp + z[p, i]
            z[j, i+1] = z[j, i] + diffs[i] *\
                        (-params[nd-j] * l**((nf+j+1)/(ord+1)) * signed_power(w[0, i], (nd-j)/(ord+1)) + temp)
        z[nd, i+1] = z[nd, i] + diffs[i] * (-params[0] * l * signed_power(w[0, i], 0))
    return z, w


sig = np.zeros((2, 100001))
sig[0] = np.linspace(0, 10, 100001)
sig[1] = 0.5 * np.sin(sig[0]) + 0.8 * np.cos(0.8 * sig[0])
noise = 3 * np.cos(10000 * sig[0]) - 6 * np.sin(20000 * sig[0]) \
        - 4 * np.cos(70000 * sig[0]) + np.random.normal(0, 0.01, 100001)
xval = sig[0]
differs1, filters1 = differentiator(sig[1], 5, 2, 0.0001, 1, 10, 100001)
differs2, filters2 = differentiator(sig[1] + noise, 5, 2, 0.0001, 1, 10, 100001)
for i in range(6):
    plt.plot(xval, differs1[i])
    plt.plot(xval, differs2[i])
    plt.show()
