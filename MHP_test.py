from MHP import MHP
import numpy as np
N = 120
I = 4
M = 6
n = 8
mu_triangle = 0.01
alpha_triangle = 0.5
mu = mu_triangle * np.random.rand(M) + 0.5 * mu_triangle
#print(mu)
alpha = alpha_triangle * np.random.rand(M) + 0.5 * alpha_triangle
# print(alpha)
# print(alpha[0][1])
# print(alpha[1][1])
w = 1

P = MHP(mu = mu, alpha = alpha, omega = w, I = I, M = M, N = N)
P.actor_p()
P.generate_seq()
data_copy = P.data
len_data = len(data_copy)
rand_len = np.random.randint(len_data, size = 8)
for rand_num in rand_len:
    data_copy[rand_num][1] = -1

print('generate done!')


P.EM(Ahat = alpha, mhat = mu, omega = w, seq = data_copy, N = N, M = M, maxiter=10000)


# P.plot_events()
# P.plot_rates()


