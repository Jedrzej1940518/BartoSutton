"""
Exercise 2.5 (programming) Design and conduct an experiment to demonstrate the
diculties that sample-average methods have for nonstationary problems. Use a modified
version of the 10-armed testbed in which all the q⇤(a) start out equal and then take
independent random walks (say by adding a normally distributed increment with mean
zero and standard deviation 0.01 to all the q⇤(a) on each step). Prepare plots like
Figure 2.2 for an action-value method using sample averages, incrementally computed,
and another action-value method using a constant step-size parameter, ↵ = 0.1. Use
" = 0.1 and longer runs, say of 10,000 steps.

"""
from numpy import random
from multi_armed_bandit import e_greedy

import matplotlib.pyplot as plt

def modify_qs(mean, sd, q_a):
    return [q + random.normal(loc = mean, scale = sd) for q in q_a]

def plot_rewards(steps, avg_rewards, const_rewards):

    index = [i for i in range(0, steps)]

    plt.plot(index, avg_rewards, label='Optimal Average Rewards', marker='o')
    plt.plot(index, const_rewards, label='Optimal Constant Rewards', marker='x')

    plt.title('Optimal Average Rewards vs Optimal Constant Rewards')
    plt.xlabel('Index')
    plt.ylabel('Rewards')

    plt.legend()

    plt.show()
  
def main():
    
    
    k = 10
    q_a = [0] * k

    mean = 0
    sd = 0.01
    runs = 50000
    e = 0.1
    a = 0.1
    variance = 1
    q_t_avg = [0] * k
    q_t_ns = [0] * k

    q_t_const = [0] * k
    q_a = modify_qs(mean, sd, q_a)
    
    a_avg = lambda i: 1/(q_t_ns[i] +1)
    a_const = lambda i: a

    avg_rewards = [0]
    const_rewards = [0]
    
    avg_optimals = [0]
    const_optimals = [0]

    for i in range(runs):
        [n, avg_reward] = e_greedy(e, a_avg, k, q_t_avg, q_a, variance)
        q_t_ns[n]+=1
        
        [n_c, const_reward] = e_greedy(e, a_const, k, q_t_const, q_a, variance)
        optimal_i = q_a.index(max(q_a))
    
        q_a = modify_qs(mean, sd, q_a)
        
        if i == 0:
            continue
        avg_rewards.append(avg_rewards[-1] + 1/i *(avg_reward - avg_rewards[-1]))
        const_rewards.append(const_rewards[-1] + 1/i *(const_reward - const_rewards[-1]))

        avg_optimals.append(avg_optimals[-1] + 1/i *(int(n==optimal_i) - avg_optimals[-1]))
        const_optimals.append(const_optimals[-1] + 1/i *(int(n_c==optimal_i) - const_optimals[-1]))
        
    plot_rewards(runs, avg_optimals, const_optimals)
 
    pass
        
main()