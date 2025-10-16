from .base_optimiser import Optimiser
import numpy as np

class CellularAutomataOptimiser(Optimiser):
    def __init__(self, L=5, mu=0.5, omega=0.5):
        super().__init__()
        self.L = L
        self.mu = mu # used for classifying cells as good or bad
        self.omega = omega # used to calculate the tolerance bounds to consider a good cell or not in the averaging.

    def _initialise_lattice(self, model):
            bounds = model.get_param_bounds()
            N_params = len(bounds)

            P_lattice = np.zeros((self.L, self.L, N_params)) # this will create a grid of L x L (e.g 9 x 9), and each cell in the grid will have a vector of size N_params e.g. 3
            for i in range(self.L): # denotes the row
                for j in range(self.L): # denotes the column
                    initial_params = []
                    for k in range(N_params): # denotes the vector stored in cell (i, j)
                        p_min, p_max = bounds[k] # bounds is essentially [(-5, 5), (-5, 5)], so bounds[0] will be just the tuple (-5, 5)
                        initial_params.append(np.random.uniform(p_min, p_max))
                    P_lattice[i, j] = np.array(initial_params)
            
            loss_lattice = np.zeros((self.L, self.L))

            return P_lattice, loss_lattice

    def _evaluate_fitness(self, model, loss_function, X, y, P_lattice, loss_lattice):
                        
            for i in range(self.L):
                for j in range(self.L):
                    current_params = P_lattice[i, j]
                    model.set_params(current_params)

                    y_pred = model.predict(X)
                    loss = loss_function.compute_loss(y, y_pred)

                    loss_lattice[i, j] = loss

            F_min = np.min(loss_lattice)
            F_max = np.max(loss_lattice)
            min_index = np.unravel_index(np.argmin(loss_lattice), loss_lattice.shape)
            P_best = P_lattice[min_index]

            return loss_lattice, F_min, F_max, P_best


    def _classify_cells(self, loss_lattice, F_min, F_max):
        numerator = loss_lattice - F_min
        denomenator = F_max - F_min
        epsilon = 1e-10 # prevent division by zero

        # note: we are checking the cells for the normalised relative loss and not the actual loss
        normalised_relative_fitness = numerator / (denomenator + epsilon)
        is_good_cell = (normalised_relative_fitness <= self.mu)

        return is_good_cell, normalised_relative_fitness
    
    def _tolerance(self, model):
         bounds = model.get_param_bounds()
         a, b = bounds[0]
         return self.omega * (b - a)
    
    def _exploit_params(self, model, loss_function, X, y, P_lattice, loss_lattice, F_min, F_max):
        is_good, _ = self._classify_cells(loss_lattice, F_min, F_max)
        P_new_lattice = P_lattice.copy()
        tolerance_width = self._tolerance(model)

        for i in range(self.L):
            for j in range(self.L):
                if is_good[i, j]:
                    i_start = max(0, i - 1)
                    i_end = min(self.L, i + 2)
                    j_start = max(0, j - 1)
                    j_end = min(self.L, j + 2)

                    center_params = P_lattice[i, j].copy()
                    new_params = center_params.copy()
                    current_loss = loss_lattice[i, j]

                    # Process EACH parameter independently
                    for param_idx in range(len(center_params)):
                        # Collect good neighbor values for THIS parameter only
                        good_neighbor_vals = []
                        for row in range(i_start, i_end):
                            for col in range(j_start, j_end):
                                if row == i and col == j:
                                    continue
                                if is_good[row, col]:
                                    neighbor_val = P_lattice[row, col, param_idx]
                                    # Check tolerance for this parameter
                                    lower = center_params[param_idx] - tolerance_width
                                    upper = center_params[param_idx] + tolerance_width
                                    if lower < neighbor_val < upper:
                                        good_neighbor_vals.append(neighbor_val)

                        # Average this parameter with good neighbors
                        if good_neighbor_vals:
                            all_vals = [new_params[param_idx]] + good_neighbor_vals
                            candidate_val = np.mean(all_vals)
                            
                            # Create candidate with just this parameter changed
                            candidate_params = new_params.copy()
                            candidate_params[param_idx] = candidate_val

                            # Evaluate
                            model.set_params(candidate_params)
                            y_pred = model.predict(X)
                            candidate_loss = loss_function.compute_loss(y, y_pred)

                            # Only accept if better
                            if candidate_loss < current_loss:
                                new_params[param_idx] = candidate_val
                                current_loss = candidate_loss

                    P_new_lattice[i, j] = new_params

        return P_new_lattice
    

    def _explore_params(self, model, loss_function, X, y, P_lattice, loss_lattice, F_min, F_max):
        is_good, _ = self._classify_cells(loss_lattice, F_min, F_max)
        P_new_lattice = P_lattice.copy()
        bounds = model.get_param_bounds()

        for i in range(self.L):
            for j in range(self.L):
                i_start = max(0, i - 1)
                i_end = min(self.L, i + 2)
                j_start = max(0, j - 1)
                j_end = min(self.L, j + 2)

                center_params = P_lattice[i, j]
                new_params = center_params.copy()
                current_loss = loss_lattice[i, j]

                # Process each parameter independently
                for param_idx in range(len(center_params)):
                    neighbour_vals = []
                    for row in range(i_start, i_end):
                        for col in range(j_start, j_end):
                            if row == i and col == j:
                                continue
                            neighbour_vals.append(P_lattice[row, col, param_idx])

                    k = len(neighbour_vals)
                    candidate_params = new_params.copy()

                    if not is_good[i, j]:  # BAD CELL
                        if k > 0:
                            center_val = new_params[param_idx]
                            squared_diffs = sum((center_val - val)**2 for val in neighbour_vals)
                            std_dev = np.sqrt(squared_diffs / k)

                            param_range = bounds[param_idx][1] - bounds[param_idx][0]
                            max_perturbation = 0.1 * param_range
                            std_dev = min(std_dev, max_perturbation)

                            random_sign = 1 if np.random.random() > 0.5 else -1
                            candidate_params[param_idx] += random_sign * std_dev
                    else:  # GOOD CELL
                        s = np.random.uniform(0.01, 0.1)
                        random_sign = 1 if np.random.random() > 0.5 else -1
                        candidate_params[param_idx] += random_sign * s

                    # Clip to bounds
                    p_min, p_max = bounds[param_idx]
                    candidate_params[param_idx] = np.clip(candidate_params[param_idx], p_min, p_max)

                    # Evaluate and only accept if better
                    model.set_params(candidate_params)
                    y_pred = model.predict(X)
                    candidate_loss = loss_function.compute_loss(y, y_pred)

                    if candidate_loss < current_loss:
                        new_params = candidate_params.copy()
                        current_loss = candidate_loss
                    
                P_new_lattice[i, j] = new_params

        return P_new_lattice 


    def optimise(self, model, loss_function, X, y, max_iters=1000):

        P_lattice, loss_lattice = self._initialise_lattice(model)
        loss_lattice, F_min, F_max, P_best = self._evaluate_fitness(model, loss_function, X, y, P_lattice, loss_lattice)
        
        # print("\n")
        # print("="*50)
        # print(f"Configuration: L={self.L}, μ={self.mu}, ω={self.omega}, max_iters={max_iters}")
        # print("="*50)
        
        for iteration in range(max_iters):
            # Exploitation phase (3 iterations)
            for exploit_iter in range(3):
                P_lattice = self._exploit_params(model, loss_function, X, y, P_lattice, loss_lattice, F_min, F_max)
                loss_lattice, F_min, F_max, P_best = self._evaluate_fitness(model, loss_function, X, y, P_lattice, loss_lattice)

            # Exploration phase
            P_lattice = self._explore_params(model, loss_function, X, y, P_lattice, loss_lattice, F_min, F_max)
            loss_lattice, F_min, F_max, P_best = self._evaluate_fitness(model, loss_function, X, y, P_lattice, loss_lattice)
            
        return {"algorithm": "CellularAutomata","parameters": P_best, "best_loss": F_min}


    # def optimise(self, model, loss_function, X, y, max_iters=1000):

    #     P_lattice, loss_lattice = self._initialise_lattice(model)
    #     loss_lattice, F_min, F_max, P_best = self._evaluate_fitness(model, loss_function, X, y, P_lattice, loss_lattice)

    #     print("=" * 50)
    #     print("CELLULAR AUTOMATA OPTIMIZATION START")
    #     print("=" * 50)
    #     print(f"Configuration: L={self.L}, μ={self.mu}, ω={self.omega}, max_iters={max_iters}")
    #     print(f"Initial Best Loss: {F_min:.6f}")
        
    #     for iteration in range(max_iters):
    #         # Exploitation phase (3 iterations)
    #         for exploit_iter in range(3):
    #             P_lattice = self._exploit_params(model, loss_function, X, y, P_lattice, loss_lattice, F_min, F_max)
    #             loss_lattice, F_min, F_max, P_best = self._evaluate_fitness(model, loss_function, X, y, P_lattice, loss_lattice)

    #         # Exploration phase
    #         P_lattice = self._explore_params(model, loss_function, X, y, P_lattice, loss_lattice, F_min, F_max)
    #         loss_lattice, F_min, F_max, P_best = self._evaluate_fitness(model, loss_function, X, y, P_lattice, loss_lattice)
            
    #         # Print progress every 100 iterations
    #         if (iteration + 1) % 100 == 0:
    #             # print(f"Iteration {iteration + 1:4d} | Best Loss: {F_min:.6f}")
    #             param_std = np.std(P_lattice, axis=(0,1))
    #             print(f"Param std dev: {np.mean(param_std):.4f}")
    #             print(f"Params at bounds: {np.sum(np.abs(P_lattice) >= 99)}")
    #             num_good = np.sum(self._classify_cells(loss_lattice, F_min, F_max)[0])
    #             print(f"Iteration {iteration + 1:4d} | Best Loss: {F_min:.6f} | Good Cells: {num_good}/{self.L*self.L}")

    #     print("=" * 50)
    #     print("OPTIMIZATION COMPLETE")
    #     print("=" * 50)
    #     print(f"Final Best Loss: {F_min:.6f}")
    #     print(f"Best Parameters: {P_best}")
    #     print("=" * 50)
        
    #     return P_best, F_min        