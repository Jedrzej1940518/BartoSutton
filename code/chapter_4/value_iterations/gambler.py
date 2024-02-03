from value_iteration import ValueIteration

def action_space(s: int):
    return range(1, s+1)

pH = 0.4
theta = 0.1
#function that returns list of probabilities of s', r given s,a
#so p_s_r__s_a(s,a) could return [<0.1, 2, 30>, <0.9, 4, -1>]
def state_action_probabilities_mapping(s: int, a:int):
    
    p_win = pH
    p_lose = 1 - pH

    p_s_r__s_a = []
    
    win_state = min(s+a, 100)
    lose_state = max(s-a, 0)
    r_win = 1 if win_state == 100 else 0
    r_lose = 0 if lose_state == 0 else 0

    p_s_r__s_a.append([p_win, win_state, 0])
    p_s_r__s_a.append([p_lose, lose_state, 0])
    
    return p_s_r__s_a
    
def main():
    
    vi = ValueIteration(101, [0,100], theta, 1, state_action_probabilities_mapping, action_space)
    [Vs, Pis] = vi.value_iteration()
    print(Vs)
    
main()