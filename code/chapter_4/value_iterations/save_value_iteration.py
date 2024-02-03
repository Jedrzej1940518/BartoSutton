import matplotlib.pyplot as plt

def save_policy(policy, filename='policy_graph.png'):
    """
    Saves a graph of the policy.
    Args:
    - policy (dict): A dictionary mapping states to actions.
    - filename (str): Filename for the saved graph.
    """
    states = list(policy.keys())
    actions = [policy[state] for state in states]

    plt.figure(figsize=(10, 6))
    plt.plot(states, actions, marker='o')
    plt.title('Policy vs State')
    plt.xlabel('State')
    plt.ylabel('Action')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def save_sweeps(state_values, filename='value_sweeps.png'):
    """
    Saves a graph of state values after each sweep.
    Args:
    - state_values (list of dicts): A list where each element is a dictionary of state values for a particular sweep.
    - filename (str): Filename for the saved graph.
    """
    plt.figure(figsize=(10, 6))

    # Plot each sweep with a different color
    for i, sweep in enumerate(state_values):
        states = list(sweep.keys())
        values = [sweep[state] for state in states]
        color = 'black' if i == len(state_values) - 1 else None
        linestyle = '-' if i == len(state_values) - 1 else '--'
        plt.plot(states, values, color=color, linestyle=linestyle, label=f'Sweep {i+1}')

    plt.title('State Values Across Sweeps')
    plt.xlabel('State')
    plt.ylabel('State Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
