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


sig = []
with open('Noisy_Signal.txt', encoding='utf-8') as f:
    for line in f.readlines():
        sig += [[float(line.split()[0]), float(line.split()[1])]]
sig = np.transpose(np.array(sig))
xval = sig[0]
# plt.plot(xval, sig[1])
# plt.xlabel('t')
# plt.ylabel(r'f(t)')
# plt.show()
differs, filters = differentiator(sig[1], 2, 5, 1e-4, 9.6, 10, xval.size)
fig, axs = plt.subplot_mosaic([['ul', 'ur'], ['ml', 'mr'], ['b', 'b']])
axs['ul'].plot(xval, differs[0])
axs['ul'].set_ylabel(r'$w_1(t)$')
axs['ul'].set_xlabel('t')
axs['ur'].plot(xval, differs[1])
axs['ur'].set_ylabel(r'$w_2(t)$')
axs['ur'].set_xlabel('t')
axs['ml'].plot(xval, differs[2])
axs['ml'].set_ylabel(r'$w_3(t)$')
axs['ml'].set_xlabel('t')
axs['mr'].plot(xval, filters[3])
axs['mr'].set_ylabel(r'$w_4(t)$')
axs['mr'].set_xlabel('t')
axs['b'].plot(xval, filters[4])
axs['b'].set_ylabel(r'$w_5(t)$')
axs['b'].set_xlabel('t [time]')
plt.show()
# fig, axs = plt.subplots(3, 1)
# axs[0].plot(xval, filters[0])
# axs[0].set_ylabel(r'$w_1(t)$')
# # axs[0].grid()
# axs[1].plot(xval, filters[1])
# axs[1].set_ylabel(r'$w_2(t)$')
# # axs[1].grid()
# axs[2].plot(xval, filters[2])
# axs[2].set_ylabel(r'$w_3(t)$')
# axs[2].set_xlabel('t [time]')
# # axs[3].plot(xval, filters[3])
# # axs[3].set_ylabel(r'$w_4(t)$')
# # # axs[0].grid()
# # axs[4].plot(xval, filters[4])
# # axs[4].set_ylabel(r'$w_5(t)$')
# # axs[1].grid()
# axs[2].set_xlabel('t [time]')
# # axs[2].grid()
# plt.show()
# fig1, axs1 = plt.subplots(3)
calculations = differs[:, 20000:]
# axs1[0].plot(xval[20000:], calculations[0])
# axs1[0].set_ylabel(r'$z_0(t)$')
# axs1[0].set_xlim(2, 10)
# axs1[1].plot(xval[20000:], calculations[1])
# axs1[1].set_ylabel(r'$z_1(t)$')
# axs1[1].set_xlim(2, 10)
# axs1[2].plot(xval[20000:], calculations[2])
# axs1[2].set_ylabel(r'$z_2(t)$')
# axs1[2].set_xlim(2, 10)
# axs1[2].set_xlabel('t [time]')
# for i in range(5):
#     plt.figure(i)
#     plt.plot(xval, filters[i])
#     plt.title(rf'$w_{i + 1}(t)$')
# plt.plot(xval[25000:], calculations[0] ** 2 + (1/1.48176 * calculations[1]) ** 2)
# plt.show()
freq = (np.mean(np.sqrt(np.abs(calculations[2][np.abs(calculations[0]) > 0.5] / calculations[0][np.abs(calculations[0]) > 0.5]))))
a = np.mean(np.sqrt(calculations[0] ** 2 + (1/freq * calculations[1]) ** 2))
plt.plot(xval[20000:], np.sqrt(calculations[0] ** 2 + (1/freq * calculations[1]) ** 2))
plt.ylabel(r'$\sqrt{z_0^2(t)+(\frac{z_1(t)}{\omega})^2}$')
plt.xlabel('t [time]')
plt.show()
print(f'w = {freq:.5f}, a = {a:.5f}')
