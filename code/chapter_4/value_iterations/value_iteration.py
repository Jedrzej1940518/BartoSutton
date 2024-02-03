
from typing import Callable, List, Tuple
from save_value_iteration import save_policy, save_sweeps

class ValueIteration:

    def __init__(self, states: int, terminal_states: List[int], theta:float, gamma:float, state_action_probabilities_mapping:Callable[[int, int], List[Tuple[float, int, float]]], action_space: Callable[[int], List[int]]):
        
        #terminal states
        self.terminal_states = terminal_states
        #number of states including terminal states
        self.states = states
        #State values, V[1] - value of state 1
        self.V = {}
        #Policies, Pi[1] - returns action a that should be taken under state s
        self.Pi = {}
        #Accuracy of estimation, should be a small positive number        
        self.theta = theta
        #discount factor
        self.gamma = gamma
        
        #function that returns list of probabilities of s', r given s,a
        #so p_s_r__s_a(s,a) could return [<0.1, 2, 30>, <0.9, 4, -1>]
        #meaning that taking action a under state s would result in
        #10 percent of state 2 reward 30, 90 percent of state 4 reward -1 

        self.state_action_probabilities_mapping = state_action_probabilities_mapping

        #takes in s, returns array of possible actions
        self.action_space = action_space


    #returns Expected action value under state a and action a
    def q(self, s:int, a:int) -> float:
        
        probabilities_states_rewards = self.state_action_probabilities_mapping(s,a)
        
        expected_r = 0
        expected_s_p_value = 0
        
        for [p, s_p, r] in probabilities_states_rewards:
            expected_r += p * r
            expected_s_p_value += p * self.V[s_p]
        
        return expected_r + self.gamma * expected_s_p_value
    
    #returns maximizing action a under state s
    def max_a(self, s:int) ->int:
    
        a_max = 1
        q_a_max = 0 #possible bugs

        for a in self.action_space(s):
            q_a = self.q(s, a)
            if q_a > q_a_max:
                a_max = a
                q_a_max = q_a

        return [a_max, q_a_max]

    def policy_evaluation(self):

        n = 0
        delta = self.theta + 1
        sweeps = []
        while delta >= self.theta and n < 15:
           print(f'Evaluating policy, sweep {n}') 
           delta = 0
           
           for s in range(self.states):
               if s == 0 or s == 100:
                   continue           
               v = self.V[s]
               [a, q_a] = self.max_a(s)
               self.V[s] = q_a
               delta = max(delta, abs(v - q_a))
           
           sweeps.append(self.V)
           n+=1
        
        save_sweeps(sweeps)
        pass
                
            
          
    def determine_best_policy(self):
        for s in range(self.states):
            self.Pi[s] = self.max_a(s)

        save_policy(self.Pi)
            

    #returns state values V[] and policies
    def value_iteration(self):
    
        #1. Initialization
        for i in range(self.states):
            self.V[i] = 0
            self.Pi[i] = 1
            

        self.V[0] = 0
        self.V[100] = 1

        #policy evaluation / sweep
        print("yao")
        self.policy_evaluation()
        print("yo")
        self.determine_best_policy()
        print("yox")

        return [self.V, self.Pi]