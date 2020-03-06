from MHP import MHP
import numpy as np
import pandas as pd
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
pd_data = pd.DataFrame(P.data)
pd_data.to_csv('pd_data.csv')
len_data = len(data_copy)
rand_len = np.random.randint(len_data, size = 8)
for rand_num in rand_len:
    data_copy[rand_num][1] = -1
pd_data_copy = pd.DataFrame(data_copy)
pd_data_copy.to_csv('pd_data_copy.csv')

print('generate done!')


ahat_copy, mhat_copy, pi_copy = P.EM(Ahat = alpha, mhat = mu, omega = w, seq = data_copy, N = N, M = M, maxiter=10000)
pd_pi_copy = pd.DataFrame(pi_copy)
pd_pi_copy.to_csv('pd_pi_copy.csv')
print('simulation done!')





