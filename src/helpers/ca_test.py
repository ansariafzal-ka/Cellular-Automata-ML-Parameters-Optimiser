import numpy as np

class UpdateParameters:
    def __init__(self, grid_size, parameters_range, objective_function, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.parameters_range = parameters_range
        self.objective_function = objective_function
        self.objective_values = None
        self.grid_size = grid_size
        self.grid = None
        self.best_param = None
        self.best_value = None

    def initialise_grid(self):
        min_val, max_val = self.parameters_range
        self.grid = np.random.randint(min_val, max_val, (self.grid_size, self.grid_size))

    def evaluate_objective_function(self):
        self.objective_values = objective_function(self.grid)

    def find_global_best(self):
        min_index = np.argmin(self.objective_values)
        min_position = np.unravel_index(min_index, self.objective_values.shape)
        
        self.best_param = self.grid[min_position]
        self.best_value = self.objective_values[min_position]

        return self.best_param, self.best_value
    
    def update_parameters(self):    
        self.grid = self.grid + self.learning_rate * (self.best_param - self.grid)
        return self.grid

    def get_grid(self):
        return self.grid
    
    def get_objective_values(self):
        return self.objective_values


def objective_function(x):
    return (x - 5) ** 2

algo = UpdateParameters(grid_size=5, parameters_range=[0, 6], objective_function=objective_function)

algo.initialise_grid()
print(f"grid:\n{algo.get_grid()}")

algo.evaluate_objective_function()
algo.find_global_best()

for i in range(1000):
    algo.update_parameters()

print(f"grid:\n{algo.get_grid()}")

