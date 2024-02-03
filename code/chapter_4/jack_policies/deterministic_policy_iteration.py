
from typing import Callable, List, Tuple, NewType

from jack_policy import save_policy

class DeterministicPolicyIteration:

    def __init__(self, states: int, theta:float, gamma:float, state_action_probabilities_mapping:Callable[[int, int], List[Tuple[float, int, float]]], action_space: Callable[[int], List[int]]):
        
        #number of states
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

    def policy_evaluation(self):

        delta = self.theta + 1
        
        while delta >= self.theta: 
           delta = 0
           
           for s in range(self.states):
               v = self.V[s]
               a = self.Pi[s]
               self.V[s] = self.q(s,a)
               delta = max(delta, abs(v - self.V[s]))    
            
          
    def policy_improvement(self):
        policy_stable = True
        
        for s in range(self.states):
            old_action = self.Pi[s]
            action_space = self.action_space(s)
            val_max = self.q(s, old_action)
            
            for a in action_space:
                val_a = self.q(s,a)
                if val_a > val_max:
                    val_max = val_a
                    self.Pi[s] = a
                    policy_stable = False
                    
        return policy_stable

    #returns state values V[] and policies
    def policy_iteration(self):
        
        #1. Initialization
        for i in range(self.states):
            self.V[i] = 0
            self.Pi[i] = 0
            
        policy_stable = False
        
        n = 0
        
        while not policy_stable and n < 6:
            
            save_policy(self.Pi, f'Iteration_{n}')
            
            #2 Evaluation
            print("Evaluating policy....")
            self.policy_evaluation()
            
            #3 Improvment    
            print("Improving policy....")
            policy_stable = self.policy_improvement()
            
            
            #prevention
            n = n+1
        
        return [self.V, self.Pi]