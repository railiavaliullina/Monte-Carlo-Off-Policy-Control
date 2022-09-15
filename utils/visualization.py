import plotly.express as px
import numpy as np
import pandas as pd


class Visualization(object):
    def __init__(self, cfg):
        """
        Class for visualizing policy, value as action set; run time and convergence time comparison plots.
        :param cfg: config
        """
        self.cfg = cfg

    @staticmethod
    def show_plot(off_policy_control_df, off_policy_eval_df, discarded_trajectories_df, map_size):
        # accuracy
        acc_no_correction_def = off_policy_control_df[f'acc_no_correction_{map_size}_default'].to_numpy()
        acc_with_correction_def = off_policy_control_df[f'acc_with_correction_{map_size}_default'].to_numpy()
        acc_no_correction_slippery = off_policy_control_df[f'acc_no_correction_{map_size}_slippery'].to_numpy()
        acc_with_correction_slippery = off_policy_control_df[f'acc_with_correction_{map_size}_slippery'].to_numpy()

        off_policy_control_plot_df = pd.DataFrame()
        off_policy_control_plot_df['accuracy'] = list(acc_no_correction_def) + list(acc_with_correction_def) + \
                                                 list(acc_no_correction_slippery) + list(acc_with_correction_slippery)
        off_policy_control_plot_df['color'] = \
            ['no_correction_default'] * len(acc_no_correction_def) + \
            ['with_correction_default'] * len(acc_with_correction_def) + \
            ['no_correction_slippery'] * len(acc_no_correction_slippery) + \
            ['with_correction_slippery'] * len(acc_with_correction_slippery)

        steps = list(np.arange(0, len(acc_no_correction_def) * 100, 100))
        off_policy_control_plot_df['step'] = steps * 4

        fig = px.line(off_policy_control_plot_df, x='step', y='accuracy', color='color')
        fig.update_layout(title=f'Accuracies, {map_size} map')
        fig.show()

        # bias/variance
        bias_ordinary_def = off_policy_eval_df[f'bias_ordinary_{map_size}_default'].to_numpy()
        var_ordinary_def = off_policy_eval_df[f'var_ordinary_{map_size}_default'].to_numpy()
        bias_ordinary_slippery = off_policy_eval_df[f'bias_ordinary_{map_size}_slippery'].to_numpy()
        var_ordinary_slippery = off_policy_eval_df[f'var_ordinary_{map_size}_slippery'].to_numpy()

        bias_weighted_def = off_policy_eval_df[f'bias_weighted_{map_size}_default'].to_numpy()
        var_weighted_def = off_policy_eval_df[f'var_weighted_{map_size}_default'].to_numpy()
        bias_weighted_slippery = off_policy_eval_df[f'bias_weighted_{map_size}_slippery'].to_numpy()
        var_weighted_slippery = off_policy_eval_df[f'var_weighted_{map_size}_slippery'].to_numpy()

        biases_df = pd.DataFrame()
        biases_df['bias'] = list(bias_ordinary_def) + list(bias_ordinary_slippery) + \
                            list(bias_weighted_def) + list(bias_weighted_slippery)
        biases_df['color'] = \
            ['ordinary_sampling_default'] * len(bias_ordinary_def) + \
            ['ordinary_sampling_slippery'] * len(bias_ordinary_slippery) + \
            ['weighted_sampling_default'] * len(bias_weighted_def) + \
            ['weighted_sampling_slippery'] * len(bias_weighted_slippery)

        biases_df['step'] = list(np.arange(len(bias_ordinary_def))) + \
                            list(np.arange(len(bias_ordinary_slippery))) + \
                            list(np.arange(len(bias_weighted_def))) + \
                            list(np.arange(len(bias_weighted_slippery)))

        fig = px.scatter(biases_df, x='step', y='bias', color='color')
        fig.update_layout(title=f'Biases, {map_size} map')
        fig.show()

        vars_df = pd.DataFrame()
        vars_df['variance'] = list(var_ordinary_def) + list(var_ordinary_slippery) + \
                              list(var_weighted_def) + list(var_weighted_slippery)
        vars_df['color'] = \
            ['ordinary_sampling_default'] * len(var_ordinary_def) + \
            ['ordinary_sampling_slippery'] * len(var_ordinary_slippery) + \
            ['weighted_sampling_default'] * len(var_weighted_def) + \
            ['weighted_sampling_slippery'] * len(var_weighted_slippery)

        vars_df['step'] = list(np.arange(len(var_ordinary_def))) + \
                          list(np.arange(len(var_ordinary_slippery))) + \
                          list(np.arange(len(var_weighted_def))) + \
                          list(np.arange(len(var_weighted_slippery)))

        fig = px.scatter(vars_df, x='step', y='variance', color='color')
        fig.update_layout(title=f'Variances, {map_size} map')
        fig.show()

        # discarded trajectories
        with_correction_def = discarded_trajectories_df[f'with_correction_{map_size}_default'].to_numpy()
        no_correction_def = discarded_trajectories_df[f'with_correction_{map_size}_default'].to_numpy()
        with_correction_slippery = discarded_trajectories_df[f'with_correction_{map_size}_slippery'].to_numpy()
        no_correction_slippery = discarded_trajectories_df[f'with_correction_{map_size}_slippery'].to_numpy()

        discarded_tr_df = pd.DataFrame()

        discarded_tr_df['discarded_trajectories'] = list(with_correction_def) + list(no_correction_def) + \
                                                    list(with_correction_slippery) + list(no_correction_slippery)
        discarded_tr_df['color'] = \
            ['ordinary_sampling_default'] * len(with_correction_def) + \
            ['ordinary_sampling_slippery'] * len(no_correction_def) + \
            ['weighted_sampling_default'] * len(with_correction_slippery) + \
            ['weighted_sampling_slippery'] * len(no_correction_slippery)

        discarded_tr_df['step'] = list(np.arange(len(with_correction_def))) + \
                                  list(np.arange(len(no_correction_def))) + \
                                  list(np.arange(len(with_correction_slippery))) + \
                                  list(np.arange(len(no_correction_slippery)))

        fig = px.scatter(discarded_tr_df, x='step', y='discarded_trajectories', color='color')
        fig.update_layout(title=f'Discarded trajectories, Off policy control, {map_size} map')
        fig.show()

    def plot_scatters(self):
        """
        Plots bias/variance comparison plots.
        Plots will be visualized in browser with ability to zoom in and will be saved as files.
        """
        off_policy_control_df = pd.read_pickle('../data/off_policy_control_df.pickle')
        off_policy_eval_df = pd.read_pickle('../data/off_policy_eval_df.pickle')
        discarded_trajectories_df = pd.read_pickle('../data/discarded_trajectories_df.pickle')

        self.show_plot(off_policy_control_df, off_policy_eval_df, discarded_trajectories_df, map_size='small')
        self.show_plot(off_policy_control_df, off_policy_eval_df, discarded_trajectories_df, map_size='huge')
