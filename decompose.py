import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import os

# To do
# + 1. Add obligated WT column to p_clon (all 0)
# + 2. Sum identical subpopulations
# - 3. Regularize sum freq = 100 +/- error

class ClonalDecomposition:
    def __init__(self, f_var, n_subpop, lambda_sumclon, lambda_maxfreq, var_prob, dirichlet_papam, alpha, epsilon):
        # Frequency of variants
        self.f_var = f_var
        # Number of subpopulations
        self.n_subpop = n_subpop
        # Number of observed variants
        self.n_variants = self.f_var.shape[0]
        # Number of samples
        self.n_samples = self.f_var.shape[1]
        # Regularization parameter for controlling that sum of clonal frequencies not greater than 100%
        self.lambda_sumclon = lambda_sumclon
        # Regularization for keeping frequencies in the [0 ,100]% interval
        self.lambda_maxfreq = lambda_maxfreq
        # Matrix of presence of variants
        self.p_clon = self.p_clon_init(var_prob)
        # Mxtrix of clonal frequencies
        self.f_clon = self.f_clon_init(dirichlet_papam)
        # Learning rate
        self.alpha = alpha
        # Gradient calculation step
        self.epsilon = epsilon
        # Probability of draw 1 in variant
        self.var_prob = var_prob

    # Rectified linear unit (ReLu)
    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    # Cost function
    def cost(self, p_clon, f_clon):
        # Square error
        se = np.sum(np.power(p_clon.dot(f_clon) - self.f_var, 2))

        # Regularization for sum of clonal frequencies should not extend 100%
        reg_sumclon = self.lambda_sumclon * np.sum(
            np.array(
                list(
                    map(
                        self.relu,
                        np.sum(f_clon, axis=0) - 1
                    )
                )
            )
        )

        # Regularization that frequency can't be greater 100%
        reg_fclon_constr = self.lambda_maxfreq * (np.sum(f_clon[f_clon > 1]) ** 2 + np.sum(f_clon[f_clon < 0]) ** 2)
        return se, se + reg_sumclon + reg_fclon_constr

    # Initialize P_clon matrix
    def p_clon_init(self, prob):
        print('Initialize Presence of variants matrix')
        return np.random.choice([0, 1], (self.n_variants, self.n_subpop), p=[1-prob, prob])

    # Initialize F_clon matrix
    def f_clon_init(self, dirichlet_papam):
        print('Initialize Sub-population frequency matrix matrix')
        f_clon = np.zeros((self.n_subpop, self.n_samples))
        for i in range(self.n_samples):
            f_clon[:, i] = np.random.dirichlet(np.ones(self.n_subpop) * dirichlet_papam, size=1)

        return f_clon

    def p_clon_update(self):
        _, init_cost = self.cost(self.p_clon, self.f_clon)
        p_clon_new = np.zeros(self.p_clon.shape)

        for x, y in np.ndindex(self.p_clon.shape):
            p_clon_temp = self.p_clon.copy()
            # Switch 1/0 in one element
            p_clon_temp[x, y] = np.logical_not(p_clon_temp[x, y])
            _, upd_cost = self.cost(p_clon_temp, self.f_clon)

            if init_cost > upd_cost:
                p_clon_new[x, y] = np.logical_not(self.p_clon[x, y])
            else:
                p_clon_new[x, y] = self.p_clon[x, y]

        # If all elements will be 0 the algorithm will stuck. Better to reinitialize.
        if np.sum(p_clon_new) == 0:
            p_clon_new = self.p_clon_init(self.var_prob)

        return p_clon_new

    def f_clon_update(self):
        # gradient
        f_clon_grad = np.zeros(self.f_clon.shape)

        for x, y in np.ndindex(self.f_clon.shape):
            f_clon_m = self.f_clon.copy()
            f_clon_p = self.f_clon.copy()

            f_clon_m[x, y] -= self.epsilon
            f_clon_p[x, y] += self.epsilon

            _, cost_p = self.cost(self.p_clon, f_clon_p)
            _, cost_m = self.cost(self.p_clon, f_clon_m)

            f_clon_grad[x, y] = (cost_p - cost_m) / (2 * self.epsilon)

        f_clon_new = self.f_clon - self.alpha * f_clon_grad
        return f_clon_new

    def collapse_identical(self):
        # Sum identical subpopulations
        drop_subpop = set()

        for subpop in range(self.p_clon.shape[1]):
            for subpop1 in range(subpop + 1, self.p_clon.shape[1]):
                if np.min(self.p_clon[:, subpop] == self.p_clon[:, subpop1]):
                    self.f_clon[subpop, :] += self.f_clon[subpop1, :]
                    self.f_clon[subpop1, :] = 0
                    drop_subpop.add(subpop1)

        self.p_clon = np.delete(self.p_clon, list(drop_subpop), axis=1)
        self.f_clon = np.delete(self.f_clon, list(drop_subpop), axis=0)

        # remove negative proportions
        self.f_clon[self.f_clon < 0] = 0
        return 0

    # Add WT if not present and add up to 100%
    # Should be called after collapse identical
    def populate_wt(self):
        wt_index = -1

        for subpop in range(decompose.p_clon.shape[1]):
            # Check if it is wild-type
            if np.min(self.p_clon[:, subpop] == np.zeros((self.p_clon.shape[0], 1))):
                wt_index = subpop

        # sum up WT to 100%
        if wt_index >= 0:
            self.f_clon[wt_index, :] += np.array(list(map(self.relu, 1 - np.sum(self.f_clon, axis=0))))
        else:
            self.p_clon = np.concatenate((self.p_clon, np.zeros((self.p_clon.shape[0], 1))), axis=1)
            self.f_clon = np.concatenate((self.f_clon, np.zeros((1, self.f_clon.shape[1]))), axis=0)
            self.f_clon[self.f_clon.shape[0] - 1, :] += np.array(list(map(self.relu, 1 - np.sum(self.f_clon, axis=0))))
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''
       Performs observed variant frequency matrix decomposition
       into variant presence binary matrix and subpopulation frequency
       matrix using gradient descend algorithm.
    ''')

    parser.add_argument('-t', '--var_table', type=str,
                        help='Variant table. Columns: Chrom, Pos, Ref, Alt and sample columns')
    parser.add_argument('-p', '--perc2prop', action='store_true',
                        help='If frequencies are in percents bring them '
                             'to proportions. Default: False')
    parser.add_argument('--lambda_sumclon', type=float, default=4,
                        help='Regularization parameter for controlling that sum of '
                             'clonal frequencies not greater than 100. Default: 4')
    parser.add_argument('--lambda_maxfreq', type=float, default=4,
                        help='Regularization for keeping frequencies in the [0 ,100]'
                             'interval. Default: 4')
    parser.add_argument('-v', '--var_prob', type=float, default=0.1,
                        help='Probability to draw 1 (presence of variant in sub-population)'
                             ' at the initialization process. Default: 0.1')
    parser.add_argument('--dirichlet_papam', type=float, default=0.1,
                        help='Parameter for Dirichlet distribution. Default: 0.1')
    parser.add_argument('-i', '--iterations', type=int, default=500,
                        help='Number of iterations for gradient descend. Default: 500')
    parser.add_argument('-s', '--n_subpop', type=int, default=0,
                        help='Estimated number of sub-populations.'
                             'Default: 2x of observed variants.')
    parser.add_argument('-a', '--alpha', type=float, default=0.005,
                        help='Learning rate. Default: 0.005')
    parser.add_argument('-e', '--epsilon', type=float, default=10**-3,
                        help='Step for gradinet calculation. Default: 10^-3')
    parser.add_argument('-o', '--out_dir', type=str, default='output',
                        help='Output directory. Default: output')
    args = parser.parse_args()

    try:
        os.makedirs(args.out_dir)
    except OSError:
        print(f'Can\'t create output directory {args.out_dir}')
        exit(1)

    # read variant frequency table
    freq_var = pd.read_csv(args.var_table, sep='\t')
    freq_var_columns = freq_var.columns.to_list()
    sample_columns = [column for column in freq_var_columns if column not in ['Chrom', 'Pos', 'Ref', 'Alt']]

    # Convert percentages to proportions if perc2prop flag is set
    if args.perc2prop:
        freq_var[sample_columns] = freq_var[sample_columns] / 100

    freq_var_matrix = freq_var[sample_columns].to_numpy()

    n_subpop = args.n_subpop
    if not n_subpop:
        n_subpop = 2 * freq_var_matrix.shape[1]

    # Set ClonalDecomposition object
    decompose = ClonalDecomposition(
        f_var=freq_var_matrix,
        n_subpop=n_subpop,
        lambda_sumclon=args.lambda_sumclon,
        lambda_maxfreq=args.lambda_maxfreq,
        var_prob=args.var_prob,
        dirichlet_papam=args.dirichlet_papam,
        alpha=args.alpha,
        epsilon=args.epsilon
    )

    # Start gradient descend
    print('Start gradient descend')

    # Cost history
    cost_history = pd.DataFrame({
        'iteration': [i for i in range(args.iterations + 1)],
        'SE': np.zeros(args.iterations + 1),
        'Cost': np.zeros(args.iterations + 1),
        'Presence_change': np.zeros(args.iterations + 1),
        'Subpop_change': np.zeros(args.iterations + 1)
    })

    sq_error, cost = decompose.cost(decompose.p_clon, decompose.f_clon)
    cost_history.loc[0, 'SE'] = sq_error
    cost_history.loc[0, 'Cost'] = cost

    for i in range(args.iterations):
        p_clon_temp = decompose.p_clon_update()
        f_clon_temp = decompose.f_clon_update()
        cost_history.loc[i + 1, 'Presence_change'] = np.sum(np.power(decompose.p_clon - p_clon_temp, 2))
        cost_history.loc[i + 1, 'Subpop_change'] = np.sum(np.power(decompose.f_clon - f_clon_temp, 2))
        decompose.p_clon = p_clon_temp
        decompose.f_clon = f_clon_temp
        sq_error, cost = decompose.cost(decompose.p_clon, decompose.f_clon)
        cost_history.loc[i+1, 'SE'] = sq_error
        cost_history.loc[i+1, 'Cost'] = cost
        print(f'Iteration: {i+1}: SE = {sq_error}, '
              f'Cost = {cost}')

    pd.DataFrame(decompose.p_clon).to_csv(os.path.join(args.out_dir, 'presence_sub.tsv'), header=False, index=False, sep='\t')
    pd.DataFrame(decompose.f_clon).to_csv(os.path.join(args.out_dir, 'freq_sub.tsv'), header=False, index=False, sep='\t')
    pd.DataFrame(decompose.p_clon.dot(decompose.f_clon)).to_csv(os.path.join(args.out_dir, 'reconstructed.tsv'), header=False, index=False, sep='\t')


    decompose.collapse_identical()
    decompose.populate_wt()

    # Plots
    cost_history.plot(x='iteration', y=['SE', 'Cost'])
    plt.savefig(os.path.join(args.out_dir, 'Cost.pdf'))
    cost_history.plot(x="iteration", y='Presence_change')
    plt.savefig(os.path.join(args.out_dir, 'Presence_change.pdf'))
    cost_history.plot(x='iteration', y='Subpop_change')
    plt.savefig(os.path.join(args.out_dir, 'Subpop_change.pdf'))

    cost_history.to_csv(os.path.join(args.out_dir, 'Cost.tsv'), sep='\t', index=False)
    pd.DataFrame(decompose.p_clon).to_csv(os.path.join(args.out_dir, 'presence_sub_collapsed.tsv'), header=False, index=False, sep='\t')
    pd.DataFrame(decompose.f_clon).to_csv(os.path.join(args.out_dir, 'freq_sub_collapsed.tsv'), header=False, index=False, sep='\t')
    pd.DataFrame(decompose.p_clon.dot(decompose.f_clon)).to_csv(os.path.join(args.out_dir, 'reconstructed_collapsed.tsv'), header=False, index=False, sep='\t')
