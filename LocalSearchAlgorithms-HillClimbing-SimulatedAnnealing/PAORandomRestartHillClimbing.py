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

# Generate neighbors
def generate_neighbors(solution, step_size, range_min, range_max):
    neighbors = []
    for i in range(len(solution)):
        neighbor = solution[:]
        neighbor[i] += step_size
        neighbor[i] = max(min(neighbor[i], range_max), range_min)  # Clipping
        neighbors.append(neighbor)
        
        neighbor = solution[:]
        neighbor[i] -= step_size
        neighbor[i] = max(min(neighbor[i], range_max), range_min)  # Clipping
        neighbors.append(neighbor)
    return neighbors

# Hill Climbing algorithm
def hill_climbing(equations, range_min, range_max, initial_step_size, max_iterations):
    # Initialize solution
    solution = [random.uniform(range_min, range_max) for _ in range(len(equations[0]) - 1)]
    current_cost = cost_function(solution, equations)
    step_size = initial_step_size
    
    for iteration in range(max_iterations):
        neighbors = generate_neighbors(solution, step_size, range_min, range_max)
        best_neighbor = min(neighbors, key=lambda x: cost_function(x, equations))
        best_cost = cost_function(best_neighbor, equations)
        
        if best_cost < current_cost:
            solution = best_neighbor
            current_cost = best_cost
            step_size *= 1.05  # Increase step size dynamically
        else:
            step_size *= 0.95  # Decrease step size dynamically
    
    return solution, current_cost

# Random Restart Hill Climbing algorithm
def random_restart_hill_climbing(equations, range_min, range_max, initial_step_size, max_iterations, restarts):
    best_solution = None
    best_cost = float('inf')
    
    for _ in range(restarts):
        solution, cost = hill_climbing(equations, range_min, range_max, initial_step_size, max_iterations)
        if cost < best_cost:
            best_solution = solution
            best_cost = cost
    
    return best_solution, best_cost

# Main function
if __name__ == "__main__":
    equations = read_equations('coefficient.txt')
    range_min, range_max = -1000, 1000
    initial_step_size = 1.0
    max_iterations = 1000
    restarts = 10
    
    start_time = time.time()
    solution, cost = random_restart_hill_climbing(equations, range_min, range_max, initial_step_size, max_iterations, restarts)
    end_time = time.time()
    
    print("Optimal Solution:", solution)
    print("Cost:", cost)
    print("Time Taken:", end_time - start_time, "seconds")
