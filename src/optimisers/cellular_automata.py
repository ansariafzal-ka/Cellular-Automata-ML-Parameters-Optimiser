from .base_optimiser import Optimiser
import numpy as np

class CellularAutomataOptimiser(Optimiser):
    def __init__(self, L=5, mu=0.01, alpha=0.8):
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

        print(is_good_cell)

        return is_good_cell

    def _get_best_neighbour_params(self):
        pass

    def optimise(self, model, loss_function, X, y, max_iters):

        P_lattice, loss_lattice = self._initialise_lattice(model)
        loss_lattice, F_min, F_max, P_best = self._evaluate_fitness(model, loss_function, X, y, P_lattice, loss_lattice)
        print(loss_lattice)
        self._classify_cells(loss_lattice, F_min, F_max)

        return P_best, F_min
        