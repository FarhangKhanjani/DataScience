import random
import math
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

# Simulated Annealing algorithm
def simulated_annealing(equations, range_min, range_max, initial_step_size, initial_temp, cooling_rate, max_iterations):
    # Initialize solution
    solution = [random.uniform(range_min, range_max) for _ in range(len(equations[0]) - 1)]
    current_cost = cost_function(solution, equations)
    step_size = initial_step_size
    temperature = initial_temp
    
    for iteration in range(max_iterations):
        neighbor = generate_neighbor(solution, step_size, range_min, range_max)
        neighbor_cost = cost_function(neighbor, equations)
        
        # Acceptance probability
        if neighbor_cost < current_cost:
            solution = neighbor
            current_cost = neighbor_cost
        else:
            acceptance_prob = math.exp((current_cost - neighbor_cost) / temperature)
            if random.random() < acceptance_prob:
                solution = neighbor
                current_cost = neighbor_cost
        
        # Cool down the temperature
        temperature *= cooling_rate
        # Adjust step size dynamically
        step_size *= 0.99 if neighbor_cost >= current_cost else 1.01
    
    return solution, current_cost

# Main function
if __name__ == "__main__":
    equations = read_equations('coefficient.txt')
    range_min, range_max = -1000, 1000
    initial_step_size = 1.0
    initial_temp = 1000.0
    cooling_rate = 0.995
    max_iterations = 10000
    
    start_time = time.time()
    solution, cost = simulated_annealing(equations, range_min, range_max, initial_step_size, initial_temp, cooling_rate, max_iterations)
    end_time = time.time()
    
    print("Optimal Solution:", solution)
    print("Cost:", cost)
    print("Time Taken:", end_time - start_time, "seconds")
