from .base_optimiser import Optimiser
import numpy as np

class CellularAutomataOptimiser(Optimiser):
    def __init__(self, L=5, mu=0.01, alpha=0.8, omega=0.01):
        super().__init__()
        self.L = L
        self.mu = mu
        self.alpha = alpha

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

        return is_good_cell

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
    
    def _explore_params(self, current_P, P_best):
         direction = P_best - current_P
         P_new = current_P + self.alpha * direction

         return P_new

    def optimise(self, model, loss_function, X, y, max_iters=1000):
        # T=0: INITIALIZATION AND EVALUATION
        P_lattice, loss_lattice = self._initialise_lattice(model)
        
        # Store initial state for printing
        loss_lattice, F_min, F_max, P_best = self._evaluate_fitness(model, loss_function, X, y, P_lattice, loss_lattice)
        initial_P_lattice = P_lattice.copy()
        initial_loss_lattice = loss_lattice.copy()

        # ITERATION LOOP
        for t in range(max_iters):
            # 1. Re-evaluate to get the current P_best for the entire lattice
            loss_lattice, F_min, F_max, P_best = self._evaluate_fitness(model, loss_function, X, y, P_lattice, loss_lattice)
            self._classify_cells(loss_lattice, F_min, F_max) # Classification is done, but result (is_good_cell) is unused for this test
            
            # Prepare a new lattice for updates
            P_new_lattice = P_lattice.copy()
            
            # 2. Apply Exploration Rule to ALL cells
            for i in range(self.L):
                for j in range(self.L):
                    current_P = P_lattice[i, j]
                    
                    # Apply Exploration Rule (Rule 0) to ALL cells (Good or Bad)
                    # NOTE: Clamping is omitted as per your request
                    P_new = self._explore_params(current_P, P_best) 
                    
                    P_new_lattice[i, j] = P_new

            # 3. Update Lattice for next iteration
            P_lattice = P_new_lattice
        
        # FINAL EVALUATION after all updates (max_iters)
        loss_lattice, F_min, F_max, P_best = self._evaluate_fitness(model, loss_function, X, y, P_lattice, loss_lattice)
        
        # PRINTING ONLY THE REQUESTED INFORMATION
        print("----------------------------------------------------------------")
        print(f"--- CA-OF EXPLORATION-ONLY TEST ({max_iters} Iterations) ---")
        print("----------------------------------------------------------------")
        print(f"**Initial Parameter Lattice (T=0):**\n{initial_P_lattice}")
        print("\n----------------------------------------------------------------")
        print(f"**Initial Loss Lattice (T=0):**\n{initial_loss_lattice}")
        print("\n----------------------------------------------------------------")
        print(f"**Final Parameter Lattice (T={max_iters}):**\n{P_lattice}")
        print("\n----------------------------------------------------------------")
        print(f"**Final Loss Lattice (T={max_iters}):**\n{loss_lattice}")
        print(f"\nFinal Best Loss (F_min): {F_min}")
        print("----------------------------------------------------------------")

        return P_best, F_min
        