"""
Exercise 2.5 (programming) Design and conduct an experiment to demonstrate the
diculties that sample-average methods have for nonstationary problems. Use a modified
version of the 10-armed testbed in which all the q⇤(a) start out equal and then take
independent random walks (say by adding a normally distributed increment with mean
zero and standard deviation 0.01 to all the q⇤(a) on each step). Prepare plots like
Figure 2.2 for an action-value method using sample averages, incrementally computed,
and another action-value method using a constant step-size parameter, ↵ = 0.1. Use
" = 0.1 and longer runs, say of 10,000 steps.

Exercise 2.11 (programming) Make a figure analogous to Figure 2.6 for the nonstationary
case outlined in Exercise 2.5. Include the constant-step-size "-greedy algorithm with
↵= 0.1. Use runs of 200,000 steps and, as a performance measure for each algorithm and
parameter setting, use the average reward over the last 100,000 steps.

"""
from numpy import random
from multi_armed_bandit import StochasticGradientBandit, e_greedy

import matplotlib.pyplot as plt

def modify_qs(q_a):
    mean = 0
    sd = 0.01

    return [q + random.normal(loc = mean, scale = sd) for q in q_a]



def main():
    
    k = 10
    q_a = [0] * k
    variance = 0.1
    
    runs = 200000

    ##SBG
    sgb = StochasticGradientBandit(k, 0.1)
    sgb2 = StochasticGradientBandit(k, 0.25)
    sgb3 = StochasticGradientBandit(k, 0.05)

    for t in range(runs):
        sgb.take_action(q_a, variance)
        sgb2.take_action(q_a, variance)
        sgb3.take_action(q_a, variance)
        q_a = modify_qs(q_a)

    plt.plot(sgb.rewards, label='sgb (alpha=0.1)')
    plt.plot(sgb2.rewards, label='sgb2 (alpha=0.25)')
    plt.plot(sgb3.rewards, label='sgb3 (alpha=0.05)')
    plt.xlabel('Time Steps')
    plt.ylabel('Rewards')
    plt.title('Average Rewards Over Time')
    plt.legend()  # This adds the legend to the plot
    plt.show()
    
    pass
        
main()