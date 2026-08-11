import random
import time

# Read the system of equations from a file
def read_equations(file_path):
    with open(file_path, 'r') as file:
        equations = [list(map(float, line.strip().split(','))) for line in file]
    return equations

# Cost function: sum of squared errors
def cost_function(solution, equations):
    cost = 0
    for eq in equations:
        lhs = sum(coef * sol for coef, sol in zip(eq[:-1], solution))
        rhs = eq[-1]
        cost += (lhs - rhs) ** 2
    return cost

# Generate a random neighbor
def generate_neighbor(solution, step_size, range_min, range_max):
    neighbor = solution[:]
    index = random.randint(0, len(solution) - 1)
    neighbor[index] += random.uniform(-step_size, step_size)
    neighbor[index] = max(min(neighbor[index], range_max), range_min)  # Clipping
    return neighbor

# Stochastic Hill Climbing algorithm
def stochastic_hill_climbing(equations, range_min, range_max, initial_step_size, max_iterations):
    # Initialize solution
    solution = [random.uniform(range_min, range_max) for _ in range(len(equations[0]) - 1)]
    current_cost = cost_function(solution, equations)
    step_size = initial_step_size
    
    for iteration in range(max_iterations):
        neighbor = generate_neighbor(solution, step_size, range_min, range_max)
        neighbor_cost = cost_function(neighbor, equations)
        
        if neighbor_cost < current_cost:
            solution = neighbor
            current_cost = neighbor_cost
            step_size *= 1.05  # Increase step size dynamically
        else:
            step_size *= 0.95  # Decrease step size dynamically
    
    return solution, current_cost

# Main function
if __name__ == "__main__":
    equations = read_equations('coefficient.txt')
    range_min, range_max = -1000, 1000
    initial_step_size = 1.0
    max_iterations = 10000
    
    start_time = time.time()
    solution, cost = stochastic_hill_climbing(equations, range_min, range_max, initial_step_size, max_iterations)
    end_time = time.time()
    
    print("Optimal Solution:", solution)
    print("Cost:", cost)
    print("Time Taken:", end_time - start_time, "seconds")
