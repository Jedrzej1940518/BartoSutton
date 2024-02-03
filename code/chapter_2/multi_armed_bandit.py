
from numpy import random
import math 

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

class StochasticGradientBandit:
    def __init__(self, k, alpha):
        self.r_avg_t = 0    # Average reward over time <- bias
        self.n = 0          # Number of actions
        self.h_ts = [0] * k # action preferences in _t <- bias
        self.alpha = alpha  # step size
        self.rewards = []   # all rewards (for plotting purposes)
    
    def take_action(self, q_a, variance):
        e_hts_total = sum([math.exp(h_t) for h_t in self.h_ts])
        pi_ts = [math.exp(h_t)/e_hts_total for h_t in self.h_ts]

        p = random.random()
        cumulative_probability  = 0.0
        A_t = 0
        
        for i, pi_t in enumerate(pi_ts):
            cumulative_probability +=pi_t
            if p <= cumulative_probability :
                A_t = i
                break
        
        #get reward
        R_t = bandit(q_a, A_t, variance)
        # increase action number
        self.n+=1                                             
        # update avg reward
        self.r_avg_t+= (1./self.n) * (R_t - self.r_avg_t) 
        
        # Update action preferences
        for a in range(len(self.h_ts)):
            if a == A_t:
                self.h_ts[a] += self.alpha * (R_t - self.r_avg_t) * (1 - pi_ts[a])
            else:
                self.h_ts[a] -= self.alpha * (R_t - self.r_avg_t) * pi_ts[a]

        self.rewards.append(self.r_avg_t)

        return R_t