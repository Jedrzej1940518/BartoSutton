
from deterministic_policy_iteration import *
from math import exp, factorial
from typing import Dict

# states: int, theta:float, gamma:float, 
# state_action_probabilities_mapping:Callable[[int, int], List[Tuple[float, int, float]]]
# action_space: Callable[[int], List[int]]):

#given expected, returns probabilities of n. Only gives probabilities > 0.05
def poisson_distribution(expected:float) -> Dict[int, float]:

    p = 0.5
    
    i = int(expected)
    min_p = 0.001
    
    distribution = {}
    
    while p > min_p: 
        p = (expected**i *exp(-expected)) / factorial(i)
        p = round(p, 3)
        distribution[i] = p
        i=i+1    

    i = int(expected - 1)
    p = 0.5
    while  i >= 0 and p > min_p: 
        p = (expected**i *exp(-expected)) / factorial(i)
        p = round(p, 3)
        distribution[i] = p
        i=i-1   
    
    return distribution

N = 21

#maps (s1,s2) to a unique number s
def combine_state(s1:int,s2:int)->int:
    return s1 * N + s2

#maps s to two states s1, s2  
def separate_state(s:int) -> Tuple[int, int]:
    return [s // N, s % N]

def cars_action_space(s: int) -> List[int]:
    
    [s1, s2] = separate_state(s)
    l = max(-5, -s2)
    r = min(5, s1)
    
    return range(l, r+1)

requests1 = poisson_distribution(3)
returns1 = poisson_distribution(3)
    
requests2 = poisson_distribution(4)
returns2 = poisson_distribution(2)

def cars_state_action_probabilities_mapping(s:int, a:int) ->List[Tuple[float, int, float]]:
   
    [s1, s2] = separate_state(s)
    
    p_sp_r = []
    
    for requests1_num, request1_probability in requests1.items():
        for returns1_num, returns1_probability in returns1.items():
            
            for requests2_num, request2_probability in requests2.items():
                for returns2_num, returns2_probability in returns2.items():
                    
                    s_p_1 = s1 - a
                    s_p_2 = s2 + a
                    if s_p_1 < 0 or s_p_2 < 0:
                       print(f'bruuh a {a}, sp1{s_p_1}, sp2{s_p_2}')   
                        
                    s1_cars = s_p_1
                    s2_cars = s_p_2
                    
                    s_p_1 = min(max(s1_cars - requests1_num, 0) + returns1_num, 20)
                    s_p_2 = min(max(s2_cars - requests2_num, 0) + returns2_num, 20)
                    
                    r = min(s1_cars, requests1_num) * 10
                    r += min(s2_cars, requests2_num) * 10
                    r -= abs(a)  * 2
                    
                    p = request1_probability * returns1_probability * request2_probability * returns2_probability
                    s_p = combine_state(s_p_1, s_p_2)
                    p_sp_r.append([p, s_p, r])        
                    if s_p < 0 or s_p > 440:
                        print(f'a {a}, sp1 {s_p_1}, sp2 {s_p_2}')   

    return p_sp_r

def main():
    dpi = DeterministicPolicyIteration(21*21, 0.1, 0.9, cars_state_action_probabilities_mapping, cars_action_space)
    
    [Vs, Pis] = dpi.policy_iteration()
    
    print(Pis)
    print("VS")
    print(Vs)
    

main()