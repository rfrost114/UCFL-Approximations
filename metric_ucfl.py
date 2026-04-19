import numpy as np
import cvxpy as cp
from SolutionData import SolutionData

class UCLF:
    def __init__(self, file_name : str, rounding_value : float = 1e-6, max_iters : int = 1000):
        self.file_name = file_name
        self.assignment_costs, self.opening_costs = self._parse_data_file()
        self.num_clients = self.assignment_costs.shape[0]
        self.num_facilities = self.assignment_costs.shape[1]
        self.rounding_value = rounding_value # absolute distance from 1 or zero that we round to 
        self.max_iters = max_iters

        # Primal Information
        self.x_star = None # primal assignment decision variable
        self.y_star = None # primal opening decision variable
        self.primal_objective_value = None # value of primal objective function
        self.primal_integral = False # is the primal integral

        # Dual Information
        self.v_star = None # dual value variable
        self.w_star = None # dual weight variable
        self.dual_objective_value = None # value of the dual objective

        # solution data
        self.assignments = None # Assignment solution
        self.solution = None
        self.pd_iters = 0
        

    def _parse_data_file(self) -> tuple[np.ndarray, np.ndarray]:
        assignment_cost_matrix = []
        opening_cost_vector = []
        try:
            with open(self.file_name, 'r') as f:
                for i, line in enumerate(f):
                    if i > 1:
                        line_data = [int(x) for x in line.split()]
                        opening_cost_vector.append(line_data[1])
                        assignment_cost_matrix.append(line_data[2:])
        except FileNotFoundError:
            print(f'File {self.file_name} not found')
        
        return np.array(assignment_cost_matrix), np.array(opening_cost_vector)

    def _solve_primal(self, relaxation : bool = True) -> bool:
        if relaxation:
            x = cp.Variable(self.assignment_costs.shape, nonneg=True)
            y = cp.Variable(self.num_facilities, nonneg=True)
        else:
            x = cp.Variable(self.assignment_costs.shape, boolean=True) # enforce the integer constraint if we are not solving the relaxation
            y = cp.Variable(self.num_facilities, boolean=True)


        objective = cp.Minimize(cp.sum(cp.multiply(self.assignment_costs, x)) + cp.sum(cp.multiply(self.opening_costs, y)))
        constraints = []
        constraints.append(cp.sum(x, axis=1) == 1) # each client is assigned to exactly one facility

        for i in range(self.num_facilities):
            constraints.append(x[:, i] <= y[i]) # the facility assigned to each client must be open
        
        if relaxation:
            constraints.append(y <= 1)
            constraints.append(x <= 1)
        
        problem = cp.Problem(objective, constraints)
        problem.solve()

        if problem.status == 'optimal':
            self.x_star = x.value
            self.y_star = y.value
            self.primal_objective_value = problem.value
            if relaxation:
                if self._is_integral():
                    self.primal_integral = True
            else:
                self.primal_integral = True
            return True
        else:
            print(f'Error solving primal: Problem {problem.status}')
            return False

    def _is_integral(self) -> bool:
        if self.x_star is None or self.y_star is None:
            return False
        
        x_integral, y_integral = True, True
        x_non_zeros = self.x_star[self.x_star != 0]
        y_non_zeros = self.y_star[self.y_star != 0]
        for x in x_non_zeros:
            if x < 1 - self.rounding_value and x > self.rounding_value:
                x_integral = False
                # print(f'{x=}')
                break
        for y in y_non_zeros:
            if y < 1 - self.rounding_value and y > self.rounding_value:
                # print(f'{y=}')
                y_integral = False
                break
        return x_integral and y_integral



    
    def _solve_dual(self) -> bool:
        v = cp.Variable(self.num_clients)
        w = cp.Variable(self.assignment_costs.shape, nonneg=True)

        dual_objective = cp.Maximize(cp.sum(v))
        dual_constraints = []
        for j in range(self.num_facilities):
            dual_constraints.append(cp.sum(w[:, j]) <= self.opening_costs[j])

        for i in range(self.num_clients):
            dual_constraints.append(v[i] - w[i,:] <= self.assignment_costs[i,:])
        
        dual_problem = cp.Problem(dual_objective, dual_constraints)
        dual_problem.solve()

        if dual_problem.status == 'optimal':
            self.v_star = v.value
            self.w_star = w.value
            self.dual_objective_value = dual_problem.value
            return True
        else:
            print(f'Error solving dual: Problem {dual_problem.status}')
            return False

    
    def _four_approximation(self, verbose : bool = False) -> tuple[int, dict[int : list[int]]]:
        neighbourhoods = self._get_neighbourhoods()
        outer_neighbourhoods = self._get_outer_neighbourhoods(neighbourhoods)

        uncovered_clients = list(range(self.num_clients))
        assignments = { x : [] for x in range(self.num_facilities) }

        while len(uncovered_clients):
            # choose the client with the smallest value of v
            j = min(uncovered_clients, key = lambda x : self.v_star[x])
            uncovered_clients.remove(j)
            cheapest_neighbour = min(neighbourhoods[j], key=lambda x : self.opening_costs[x])
            outer_neighbours = outer_neighbourhoods[j]

            if verbose:
                print(f'Selected client {j}')
                print(f'Facility {cheapest_neighbour} assigned clients {outer_neighbours}')
            
            for n in outer_neighbours:
                if n != j:
                    try:
                        uncovered_clients.remove(n)
                        assignments[cheapest_neighbour].append(n)
                    except ValueError:
                        pass
            assignments[cheapest_neighbour].append(j)

        return {facility : client_list for facility, client_list in assignments.items() if len(client_list)}

    
    def _get_neighbourhoods(self) -> dict[int : list[int]]:
        neighbourhoods = {}
        for client in range(self.x_star.shape[0]):
            neighbourhoods[client] = []
            for facility in range(self.x_star.shape[1]):
                if self.x_star[client, facility] > self.rounding_value:
                    neighbourhoods[client].append(facility)
        return neighbourhoods

    
    def _get_outer_neighbourhoods(self, neighbourhoods : dict[int : list[int]]) -> dict[int : list[int]]:

        outer_neighbourhoods = {}
        for client, neighbours in neighbourhoods.items():
            outer_list = set()
            for neighbour in neighbours:
                facility_column = self.x_star[:, neighbour]
                for client_index in range(len(facility_column)):
                    if facility_column[client_index] > self.rounding_value:
                        outer_list.add(client_index)
            outer_neighbourhoods[client] = list(outer_list)
        return outer_neighbourhoods



    
    def _randomized_three_approximation(self, verbose : bool = False) -> tuple[int, dict[int : list[int]]]:
        neighbourhoods = self._get_neighbourhoods()
        outer_neighbourhoods = self._get_outer_neighbourhoods(neighbourhoods)

        uncovered_clients = list(range(self.num_clients))
        assignments = { x : [] for x in range(self.num_facilities) }

        # calculate cost per client under the LP solution
        C = np.sum(np.multiply(self.x_star, self.assignment_costs), axis=1)

        while len(uncovered_clients):
            j = min(uncovered_clients, key = lambda x : self.v_star[x] + C[x])

            uncovered_clients.remove(j)

            # randomly open a neighbour of j according to the distribution induced by the primal var x
            j_neighbours = np.array(neighbourhoods[j])
            weights = np.array([self.x_star[j][k] for k in j_neighbours])

            # because we based our selection of neighbours by rounding off very small values the weight vector may not sum to 1
            # we solve this by adding whatever missing weight we need to the highest weigth neighbour 
            # the missing weight is at most (1e-6) * # facilities = (1e-4) so will likely not affect the results much
            missing_weight = 1 - np.sum(weights)

            if missing_weight != 0:
                max_weight = np.max(weights)
                for weight_index in range(weights.shape[0]):
                    if weights[weight_index] == max_weight:
                        weights[weight_index] += missing_weight
                        break
            
            chosen_neighbour = np.random.choice(j_neighbours, size=1, p=weights)[0]
            outer_neighbours = outer_neighbourhoods[j]

            if verbose:
                print(f'Selected client {j}')
                print(f'Facility {chosen_neighbour} assigned clients {outer_neighbours}')
            
            for n in outer_neighbours:
                if n != j:
                    try:
                        uncovered_clients.remove(n)
                        assignments[chosen_neighbour].append(n)
                    except ValueError:
                        pass
            assignments[chosen_neighbour].append(j)

        return {facility : client_list for facility, client_list in assignments.items() if len(client_list)}


    def _primal_dual_three_approximation(self, verbose : bool = False) -> tuple[bool, dict[int : list[int]]]:
        v = np.zeros(self.num_clients)
        w = np.zeros((self.num_clients, self.num_facilities))
        S = set(range(self.num_clients)) # the set of active clients
        T = set() # set of tight facilities
        facility_set = set(range(self.num_facilities)) # a dummy set of all facilities for use in set differences
        neighbours = {c : set() for c in range(self.num_clients)} # track the neighbouring facilities of each client
        current_iter = 0
        dual_feasible = None

        while len(S) and current_iter < self.max_iters: # there is a cut off for the maximum number of iterations
            # The goal is to find the minimum amount we can increment v 

            # first we will find the amount of increase needed to make some client neighbour a facility
            min_neighbour_increase = np.inf
            for client in S:
                for facility in facility_set.difference(neighbours[client]):
                    increase_amount = max(0, self.assignment_costs[client,facility] - v[client]) # the amount needed to make client neighbour facility
                    min_neighbour_increase = min(increase_amount, min_neighbour_increase)
            
            # next we find the amount of increase needed to make a facility constraint go tight
            tightness_increase = np.inf
            for not_tight_facility in facility_set.difference(T):
                # calculate the number of active neighbours for the facility
                active_neighbour_count = len([c for c in S if not_tight_facility in neighbours[c]])
                if active_neighbour_count > 0:
                    increase_amount = ((self.opening_costs[not_tight_facility] - np.sum(w[:,not_tight_facility])) / active_neighbour_count)
                    tightness_increase = min(tightness_increase, increase_amount)
            
            # the step we make is the minimum of the two amounts
            step_increase = min(tightness_increase, min_neighbour_increase)

            # update the dual variables by the step amount
            for client in S:
                v[client] += step_increase

                client_neighbours = neighbours[client]
                for f in client_neighbours.difference(T):
                    w[client,f] += step_increase
            
            # update the neighbour set based on the new dual values
            for c in S:
                for f in facility_set:
                    if v[c] >= self.assignment_costs[c,f] - self.rounding_value: 
                        # print(f'adding neighbour {f} to {c}')
                        neighbours[c].add(f)

            # check for tight facility constraints
            for f in facility_set.difference(T):
                if np.sum(w[:,f]) >= self.opening_costs[f] - self.rounding_value:
                    T.add(f)

            # update the set of active facilities (S)
            removals = set()
            for c in S:
                if len(neighbours[c].intersection(T)) > 0:
                    removals.add(c)
            
            for r in removals:
                S.remove(r)
            
            # check that the dual is still feasible
            facility_constraints = all([np.sum(w[:, j]) <= self.opening_costs[j] + self.rounding_value for j in range(self.num_facilities)])
            weight_constraints = all([all(v[i] - w[i,:] <= self.assignment_costs[i,:]+self.rounding_value) for i in range(self.num_clients)])
            dual_feasible = facility_constraints and weight_constraints
            current_iter += 1
        
        if verbose:
            print(f'Primal Dual terminated after {current_iter} iterations')
        self.pd_iters = current_iter

        # now we cut down the the set of tight facilities 
        T_prime = set()
        while len(T):
            f = T.pop()
            T_prime.add(f)
            # find the contributors to the facility
            contributors = [c for c in range(self.num_clients) if w[c,f] > 0]

            # we will find all facilities in T that share a contributor with f and remove them
            shared_contributors = set() 
            for facility in T:
                for contributor in contributors:
                    if w[contributor, facility] > 0:
                        shared_contributors.add(facility)
            
            for x in shared_contributors:
                T.remove(x)
        
        # finally we create the assignment by assigning each client to the closest facility in T'
        assignments = {f : [] for f in T_prime}
        T_prime = list(T_prime)

        for client in range(self.num_clients):
            cheapest_assignment = min(T_prime, key=lambda x : self.assignment_costs[client, x])
            assignments[cheapest_assignment].append(client)
        
        return dual_feasible, assignments



    def _parse_integer_solution(self) -> tuple[int, dict[int : list[int]]]:
        assignments = { x : set() for x in range(self.num_facilities) }
        for client_i in range(self.x_star.shape[0]):
            for facility_j in range(self.x_star.shape[1]):
                if np.abs(self.x_star[client_i, facility_j] - 1) < self.rounding_value:
                    assignments[facility_j].add(client_i)
        return {facility : list(client_set) for facility, client_set in assignments.items() if len(client_set)}

    def _find_solution_value(self) -> tuple:
        total_assignment_cost, total_opening_cost, total_opened = 0, 0, 0
        for final_f, final_a in self.assignments.items():
            total_opened += 1
            total_opening_cost += self.opening_costs[final_f]
            for assigned_client in final_a:
                total_assignment_cost += self.assignment_costs[assigned_client, final_f]
        return total_assignment_cost, total_opening_cost, total_opened

    def solve_instance(self, method : str ='4-approx', seed : int | None = None, verbose : bool = False) -> int:

        if seed is not None:
            np.random.seed(seed)

        if method == '4-approx':
            primal = self._solve_primal() # solve for the primal
            dual = self._solve_dual() # solve for the dual

            if not primal:
                if verbose:
                    print('Failed to solve primal - Ending run')
                return 0
            if not dual:
                if verbose:
                    print('Failed to solve dual - Ending run')
                return 0
            
            if self.primal_integral: # if the approximation solves the problem optimally 
                if verbose:
                    print('Primal is integral, stopping early')
                self.assignments = self._parse_integer_solution()
            else:
                self.assignments = self._four_approximation(verbose)
                # TODO add check for solution feasibility 

            assignment_value , opening_value, total_opened = self._find_solution_value()
            if verbose:
                print(f'Total Facilities opened: {total_opened}')
                print(f'Total Facility Opening Cost: {opening_value}')
                print(f'Total Cost Of Assignment: {assignment_value}')
                print(f'Overall Cost: {assignment_value + opening_value}')
            self.solution = SolutionData(
                self.assignments,
                assignment_value,
                opening_value,
                assignment_value + opening_value,
                total_opened
            )
            return 1
            
        elif method == 'rand-3':
            primal = self._solve_primal() # solve for the primal
            dual = self._solve_dual() # solve for the dual
            if not primal:
                if verbose:
                    print('Failed to solve primal - Ending run')
                return 0
            if not dual:
                if verbose:
                    print('Failed to solve dual - Ending run')
                return 0
            
            if self.primal_integral: # if the approximation solves the problem optimally 
                if verbose:
                    print('Primal is integral, stopping early')
                self.assignments = self._parse_integer_solution()
            else:
                self.assignments = self._randomized_three_approximation(verbose)
                # TODO add check for solution feasibility 
            
            assignment_value , opening_value, total_opened = self._find_solution_value()
            if verbose:
                print(f'Total Facilities opened: {total_opened}')
                print(f'Total Facility Opening Cost: {opening_value}')
                print(f'Total Cost Of Assignment: {assignment_value}')
                print(f'Overall Cost: {assignment_value + opening_value}')
            self.solution = SolutionData(
                self.assignments,
                assignment_value,
                opening_value,
                assignment_value + opening_value,
                total_opened
            )
            return 1
            
        elif method == 'p-d-3':
            feasible, assignment = self._primal_dual_three_approximation(verbose)
            if feasible:
                self.assignments = assignment
                assignment_value , opening_value, total_opened = self._find_solution_value()
                if verbose:
                    print(f'Total Facilities opened: {total_opened}')
                    print(f'Total Facility Opening Cost: {opening_value}')
                    print(f'Total Cost Of Assignment: {assignment_value}')
                    print(f'Overall Cost: {assignment_value + opening_value}')
                self.solution = SolutionData(
                    self.assignments,
                    assignment_value,
                    opening_value,
                    assignment_value + opening_value,
                    total_opened
                )
                return 1
            else:
                if verbose:
                    print('Dual became infeasible try adjusting rounding value')
                return 0
        else:
            ...

if __name__ == '__main__':
    test_instance = UCLF('111EuclS.txt')

    status = test_instance.solve_instance(method='p-d-3', verbose=True)
    print(test_instance.pd_iters)

    # print(test_instance.x_star)
    # print(test_instance.y_star)

    

    
