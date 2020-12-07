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


parser = argparse.ArgumentParser(description='''
   Performes observed variant frequency matrix decomposition
   into variant presence binary matrix and subpopulation frequency
   matrix using gradient descend algorithm.
''')

parser.add_argument('-t', '--var_table', help='Variant table. Columns: Chrom, Pos, Ref, Alt and sample columns')
parser.add_argument('-p', '--perc2prop', action='store_true', help='If frequencies are in percents bring them '
                                                                   'to proportions. Default: False')
parser.add_argument('--lambda_sumclon', default=4.0, help='Regularization parameter for controlling that sum of '
                                                        'clonal frequencies not greater than 100%. Default: 2.0')
parser.add_argument('--lambda_maxfreq', default=4.0, help='Regularization for keeping frequencies in the [0 ,100]% '
                                                          'interval. Default: 2.0')
parser.add_argument('-v', '--var_prob', default=0.1, help='Probability to draw 1 (presence of variant in sub-population)'
                                                          ' at the initialization process. Default: 0.1')
parser.add_argument('--dirichlet_papam', default=0.1, help='Parameter for Dirichlet distribution. Default: 0.1')
parser.add_argument('-i', '--iterations', default=500, help='Number of iterations for gradient descend. Default: 50')
parser.add_argument('-s', '--n_subpop', default=0, help='Estimated number of sub-populations.'
                                                        'Default: 2x of observed variants.')
parser.add_argument('-a', '--alpha', default=0.005, help='Learning rate. Default: 0.02')
parser.add_argument('-e', '--epsilon', default=10**-3, help='Step for gradinet calculation. Default: 10^-4')
parser.add_argument('-o', '--out_dir', default='output', help='Output directory. Default: output')
args = parser.parse_args()

#try:
#    os.makedirs(args.out_dir)
#except OSError:
#    print(f'Can\'t create output directory {args.out_dir}')
#    exit(1)

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
    'Cost': np.zeros(args.iterations + 1)
})

sq_error, cost = decompose.cost(decompose.p_clon, decompose.f_clon)
cost_history.loc[0, 'SE'] = sq_error
cost_history.loc[0, 'Cost'] = cost

for i in range(args.iterations):
    p_clon_temp = decompose.p_clon_update()
    f_clon_temp = decompose.f_clon_update()
    decompose.p_clon = p_clon_temp
    decompose.f_clon = f_clon_temp
    sq_error, cost = decompose.cost(decompose.p_clon, decompose.f_clon)
    cost_history.loc[i+1, 'SE'] = sq_error
    cost_history.loc[i+1, 'Cost'] = cost
    print(f'Iteration: {i+1}: SE = {sq_error}, '
          f'Cost = {cost}')

# Sum identical subpopulations
drop_subpop = list()

for subpop in range(decompose.p_clon.shape[1]):
    for subpop1 in range(subpop + 1, decompose.p_clon.shape[1]):
        if np.min(decompose.p_clon[:, subpop] == decompose.p_clon[:, subpop1]):
            decompose.f_clon[subpop, :] += decompose.f_clon[subpop1, :]
            decompose.f_clon[subpop1, :] = 0
            drop_subpop.append(subpop1)

decompose.p_clon = np.delete(decompose.p_clon, drop_subpop, axis=1)
decompose.f_clon = np.delete(decompose.f_clon, drop_subpop, axis=0)

cost_history.plot(x="iteration", y=["SE", "Cost"])
plt.savefig(os.path.join(args.out_dir, 'Cost.pdf'))
cost_history.to_csv(os.path.join(args.out_dir, 'Cost.tsv'), sep='\t')
pd.DataFrame(decompose.p_clon).to_csv(os.path.join(args.out_dir, 'p_clon.tsv'), header=False, index=False, sep='\t')
pd.DataFrame(decompose.f_clon).to_csv(os.path.join(args.out_dir, 'f_clon.tsv'), header=False, index=False, sep='\t')
pd.DataFrame(decompose.p_clon.dot(decompose.f_clon)).to_csv(os.path.join(args.out_dir, 'reconstructed.tsv'), header=False, index=False, sep='\t')
