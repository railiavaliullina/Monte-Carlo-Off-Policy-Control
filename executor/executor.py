import pickle

import numpy as np
import pandas as pd

from MC.MC import MC
from configs.config import cfg
from enums.enums import *
from utils.visualization import Visualization


class Executor(object):
    def __init__(self):
        """
        Class for running policy evaluation algorithms with different params several times and analysing results.
        """
        self.cfg = cfg
        np.random.seed(0)
        self.seeds = np.random.choice(int(5e5), self.cfg.runs_num, replace=False)
        self.map_names = [MapName.small.name, MapName.huge.name]
        self.action_sets = [a.name for a in ActionSet]
        self.visualization = Visualization(cfg)
        self.mc = MC(cfg)

    def write_results(self, biases_ordinary_sampling, vars_ordinary_sampling, biases_weighted_sampling,
                      vars_weighted_sampling, acc_no_correction, acc_with_correction,
                      discarded_trajectories_num_no_correct, discarded_trajectories_num_with_correct):
        """
        Writes all experiments results to pickle file.
        :param biases_ordinary_sampling:
        :param vars_ordinary_sampling:
        :param biases_weighted_sampling:
        :param vars_weighted_sampling:
        :param acc_no_correction:
        :param acc_with_correction:
        :param discarded_trajectories_num_no_correct:
        :param discarded_trajectories_num_with_correct:
        """
        off_policy_control_df = pd.DataFrame()
        off_policy_eval_df = pd.DataFrame()
        discarded_trajectories_df = pd.DataFrame()

        for m_id, map_name in enumerate(self.map_names):
            for a_id, action_set in enumerate(self.action_sets):
                off_policy_eval_df[f'bias_ordinary_{map_name}_{action_set}'] = [np.mean(
                    biases_ordinary_sampling[map_name][action_set])]

                off_policy_eval_df[f'var_ordinary_{map_name}_{action_set}'] = [np.mean(
                    vars_ordinary_sampling[map_name][action_set])]

                off_policy_eval_df[f'bias_weighted_{map_name}_{action_set}'] = [np.mean(
                    biases_weighted_sampling[map_name][action_set])]

                off_policy_eval_df[f'var_weighted_{map_name}_{action_set}'] = [np.mean(
                    vars_weighted_sampling[map_name][action_set])]

                off_policy_control_df[f'acc_no_correction_{map_name}_{action_set}'] = np.mean(
                    np.stack(acc_no_correction[map_name][action_set]), 0)

                off_policy_control_df[f'acc_with_correction_{map_name}_{action_set}'] = np.mean(
                    np.stack(acc_with_correction[map_name][action_set]), 0)

                discarded_trajectories_df[f'with_correction_{map_name}_{action_set}'] = [np.mean(
                    discarded_trajectories_num_with_correct[map_name][action_set])]

                discarded_trajectories_df[f'no_correction_{map_name}_{action_set}'] = [np.mean(
                    discarded_trajectories_num_no_correct[map_name][action_set])]

        off_policy_control_df.to_pickle('../data/off_policy_control_df.pickle')
        off_policy_eval_df.to_pickle('../data/off_policy_eval_df.pickle')
        discarded_trajectories_df.to_pickle('../data/discarded_trajectories_df.pickle')
        self.visualization.plot_scatters()

    def run_sequence_of_experiments(self):
        """
        Runs sequence of experiments with different params.
        """
        acc_no_correction, acc_with_correction = \
            {map_name: {action_set: [] for action_set in self.action_sets} for map_name in self.map_names}, \
            {map_name: {action_set: [] for action_set in self.action_sets} for map_name in self.map_names}

        biases_ordinary_sampling, biases_weighted_sampling = \
            {map_name: {action_set: [] for action_set in self.action_sets} for map_name in self.map_names}, \
            {map_name: {action_set: [] for action_set in self.action_sets} for map_name in self.map_names}

        vars_ordinary_sampling, vars_weighted_sampling = \
            {map_name: {action_set: [] for action_set in self.action_sets} for map_name in self.map_names}, \
            {map_name: {action_set: [] for action_set in self.action_sets} for map_name in self.map_names}

        discarded_trajectories_num_no_correct, discarded_trajectories_num_with_correct = \
            {map_name: {action_set: [] for action_set in self.action_sets} for map_name in self.map_names}, \
            {map_name: {action_set: [] for action_set in self.action_sets} for map_name in self.map_names}

        for m_id, map_name in enumerate(self.map_names):
            self.cfg.map_name = map_name

            for a_id, action_set in enumerate(self.action_sets):
                self.cfg.action_set = action_set

                for run_id in range(self.cfg.runs_num):
                    print(f'Run: {run_id}/{self.cfg.runs_num}')
                    np.random.seed(self.seeds[run_id])
                    self.mc = MC(self.cfg)

                    off_policy_eval_results, off_policy_control_results = self.mc.run()
                    ordinary_sampling_bias, ordinary_sampling_var, weighted_sampling_bias, weighted_sampling_var = \
                        off_policy_eval_results
                    acc_off_policy_control_no_correction, acc_off_policy_control_with_correction, \
                    discarded_trajectories_num_no_correction, discarded_trajectories_num_with_correction = \
                        off_policy_control_results

                    biases_ordinary_sampling[map_name][action_set].append(ordinary_sampling_bias)
                    vars_ordinary_sampling[map_name][action_set].append(ordinary_sampling_var)

                    biases_weighted_sampling[map_name][action_set].append(weighted_sampling_bias)
                    vars_weighted_sampling[map_name][action_set].append(weighted_sampling_var)

                    acc_no_correction[map_name][action_set].append(acc_off_policy_control_no_correction)
                    acc_with_correction[map_name][action_set].append(acc_off_policy_control_with_correction)

                    discarded_trajectories_num_no_correct[map_name][action_set].append(
                        discarded_trajectories_num_no_correction)
                    discarded_trajectories_num_with_correct[map_name][action_set].append(
                        discarded_trajectories_num_with_correction)

        self.write_results(biases_ordinary_sampling, vars_ordinary_sampling, biases_weighted_sampling,
                           vars_weighted_sampling, acc_no_correction, acc_with_correction,
                           discarded_trajectories_num_no_correct, discarded_trajectories_num_with_correct)

    def run(self):
        """
        Runs whole pipeline.
        """
        if self.cfg.run_single_exp:
            self.visualization.plot_scatters()
            self.mc.run()
        else:
            self.visualization.plot_scatters()
            self.run_sequence_of_experiments()


if __name__ == '__main__':
    executor = Executor()
    executor.run()
