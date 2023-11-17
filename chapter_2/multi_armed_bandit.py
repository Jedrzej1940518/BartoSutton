
from numpy import random

def bandit(q_a, i, variance):
    return random.normal(q_a[i], variance)

#q_a - real expected value, q_t estimated value in T
def e_greedy(e, a, k, q_t, q_a, variance):
    r = random.random()
    if(r < e):
        i = random.randint(0, k)
    else:
        i = q_t.index(max(q_t))
        
    reward = bandit(q_a, i, variance)
    q_t[i] = q_t[i] + a(i) * (reward - q_t[i])

   # print(f'r{r}, e{e}, i{i}, q_t{q_t}')
   # print(q_t)

    return [i, reward]
