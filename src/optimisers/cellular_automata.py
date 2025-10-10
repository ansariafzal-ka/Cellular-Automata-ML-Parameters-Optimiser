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
    
    def _get_best_neighbour_params(self, i, j, P_lattice, loss_lattice):
        i_start = max(0, i - 1)
        i_end = min(self.L, i + 2)
        j_start = max(0, j - 1)
        j_end = min(self.L, j + 2)

        """
        -> below gives the 3 x 3 neighbourhood loss of a cell eg :-
                    [[9.4, 1.8, 7.2],
                    [6.1, 0.9, 5.4], 
                    [3.2, 4.8, 8.9]]
        """
        neighbour_losses = loss_lattice[i_start:i_end, j_start:j_end]

        """
        -> below gives the 3 x 3 neighbourhood param vectors of a cell eg:-
                [
                    [[param1, param2], [param1, param2], [param1, param2]],   row1 neighbors
                    [[param1, param2], [param1, param2], [param1, param2]],   row2 neighbors  
                    [[param1, param2], [param1, param2], [param1, param2]]    row3 neighbors
                ]
        """
        neighbour_P = P_lattice[i_start:i_end, j_start:j_end]

        relative_min_flat_idx = np.argmin(neighbour_losses)
        relative_min_idx = np.unravel_index(relative_min_flat_idx, neighbour_losses.shape)

        P_n = neighbour_P[relative_min_idx]

        return P_n
    
    def _tolerance(self, model):
         bounds = model.get_param_bounds()
         a, b = bounds[0]
         return self.omega * (b - a)
    
    def _exploit_params(self, model, loss_function, X, y, P_lattice, loss_lattice, F_min, F_max):
         # classify cells as good or bad
        is_good, _ = self._classify_cells(loss_lattice, F_min, F_max)
        P_new_lattice = P_lattice.copy()
        tolerance_width = self._tolerance(model)

        # loop through the entire lattice
        for i in range(self.L):
            for j in range(self.L):
                # check if a cell is good, and start finding its moore neighbourhood
                if is_good[i, j]:
                    i_start = max(0, i - 1)
                    i_end = min(self.L, i + 2)
                    j_start = max(0, j - 1)
                    j_end = min(self.L, j + 2)

                    # find the good parameters in that moore neighbourhood
                    good_neighbour_params = []
                    for row in range(i_start, i_end):
                        for col in range(j_start, j_end):
                            if row == i and col == j:
                                continue
                            if is_good[row, col]:
                                good_neighbour_params.append(P_lattice[row, col])   

                    center_params = P_lattice[i, j]
                    tolerated_neighbours = []

                    for neighbour_params in good_neighbour_params:
                        all_within_tolerance = True
                        for param_idx in range(len(center_params)):
                            lower = center_params[param_idx] - tolerance_width
                            upper = center_params[param_idx] + tolerance_width

                            if not (lower < neighbour_params[param_idx] < upper):
                                all_within_tolerance = False
                                break

                        if all_within_tolerance:
                            tolerated_neighbours.append(neighbour_params)

                    if tolerated_neighbours:
                        all_params_to_average = [center_params] + tolerated_neighbours
                        new_params = np.mean(all_params_to_average, axis=0)

                        model.set_params(new_params)
                        y_pred = model.predict(X)
                        loss_candidate = loss_function.compute_loss(y, y_pred)

                        current_loss = loss_lattice[i, j]
                        if loss_candidate < current_loss:
                            P_new_lattice[i, j] = new_params

        return P_new_lattice
                        

    def optimise(self, model, loss_function, X, y, max_iters=1000):

        P_lattice, loss_lattice = self._initialise_lattice(model)
        loss_lattice, F_min, F_max, P_best = self._evaluate_fitness(model, loss_function, X, y, P_lattice, loss_lattice)


        print("\nInitial Loss Lattice:")
        print(loss_lattice)

        for exploit_iter in range(10):
            P_lattice = self._exploit_params(model, loss_function, X, y, P_lattice, loss_lattice, F_min, F_max)
    
            # Re-evaluate fitness after each exploitation
            loss_lattice, F_min, F_max, P_best = self._evaluate_fitness(model, loss_function, X, y, P_lattice, loss_lattice)
    
            print(f"\nAfter Exploitation {exploit_iter + 1}:")
            print(F_min)


        return P_best, F_min
        