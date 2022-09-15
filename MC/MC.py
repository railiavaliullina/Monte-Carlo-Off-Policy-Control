import gym
import numpy as np
import time

from utils.visualization import Visualization
from utils.data_utils import read_file


class MC(object):
    """
    Class with first/every visit policy evaluation implementation.
    """

    def __init__(self, cfg):
        """
        Initializes class.
        :param cfg: config
        """
        self.cfg = cfg
        self.visualization = Visualization(cfg)

        self.init_env()
        self.show_env_info()

        self.transition_matrix = self.env.transition_matrix
        self.states_num = len(self.transition_matrix)
        self.actions_num = len(self.transition_matrix[0])
        self.actions_space = self.env.action_space.n
        self.get_policy()
        self.init_matrices()
        self.get_model_with_policy()

    def init_env(self):
        """
        Initializes environment with parameters from config.
        """
        self.env = gym.make(f'frozen_lake:{self.cfg.env_type}', map_name=self.cfg.map_name,
                            action_set_name=self.cfg.action_set)
        self.env.reset(start_state_index=0)

    def show_env_info(self):
        """
        Prints information about current environment.
        """
        if self.cfg.verbose:
            self.env.render(object_type="environment")
            self.env.render(object_type="actions")
            self.env.render(object_type="states")
            print(f'\nMap type: {self.cfg.map_name}\n'
                  f'Policy type: {self.cfg.policy_type}\n'
                  f'Action Set: {self.cfg.action_set}\n'
                  f'Discount Factor: {self.cfg.discount_factor}')

    def init_matrices(self):
        """
        init transition probability matrix, reward matrix.
        """
        self.transition_prob_matrix = np.zeros((self.states_num, self.states_num, self.actions_num))
        self.reward_matrix = np.zeros((self.states_num, self.actions_num))

        for s in range(self.states_num):
            for a in range(self.actions_num):
                for new_s_tuple in self.transition_matrix[s][a]:
                    transition_prob, new_s, reward, _ = new_s_tuple
                    self.transition_prob_matrix[s][new_s][a] += transition_prob
                    self.reward_matrix[s][a] += reward * transition_prob

    def get_model_with_policy(self):
        """
        Gets transition probability matrix, reward matrix within chosen policy.
        :return:
        """
        self.transition_prob_matrix_pi = np.zeros((self.states_num, self.states_num))
        self.reward_matrix_pi = np.zeros(self.states_num)

        for s in range(self.states_num):
            for new_s in range(self.states_num):
                self.transition_prob_matrix_pi[s][new_s] = self.policy[s] @ self.transition_prob_matrix[s][new_s]
            self.reward_matrix_pi[s] = self.policy[s] @ self.reward_matrix[s]

    def get_direct_solution(self):
        """
        Gets direct solution with given formula V = (I − γP)^−1 R.
        :return: v vector
        """
        start_time = time.time()

        eye_matrix = np.eye(self.states_num)
        v_pi = np.linalg.inv(
            eye_matrix - self.cfg.discount_factor * self.transition_prob_matrix_pi) @ self.reward_matrix_pi

        self.direct_solution_time = time.time() - start_time
        if self.cfg.verbose:
            print(f'\nDirect solution time: {self.direct_solution_time} s')
        return v_pi

    def get_policy(self):
        """
        Initializes policy within config params.
        """
        self.policy = np.ones((self.states_num, self.actions_num)) / self.actions_num

    def get_optimal_policy(self):
        data = read_file(self.cfg.optimal_policies_file_path)
        self.optimal_policy = np.asarray(data[self.cfg.map_name][self.cfg.action_set])

    @staticmethod
    def get_bias_and_variance(direct, visit_pe):
        visit_pe = np.asarray(visit_pe)
        bias = abs(np.mean([direct - pe for pe in visit_pe]))
        var = np.mean((visit_pe ** 2)) - np.mean(visit_pe) ** 2
        return bias, var

    def generate_episode(self, policy):

        episode = []
        s = self.env.reset()

        while True:
            action = np.random.choice(np.arange(self.actions_num), 1, p=policy[s])[0]
            new_state_index, reward, done, _ = self.env.step(action)
            episode.append((s, action, reward))
            if done:
                break
            s = new_state_index
        return np.asarray(episode)

    def get_random_policy(self):
        policy = np.random.uniform(0, 1, (self.states_num, self.actions_num))
        policy = np.asarray([policy[i, :] / np.sum(policy, 1)[i] for i in range(self.states_num)])
        return policy

    def get_greedy_policy(self):
        policy = np.zeros((self.states_num, self.actions_num), dtype=float)
        policy[:, 0] = 1.0
        return policy

    def off_policy_control(self, with_correction):
        """
        Runs off policy control algorithm.
        """
        Q = np.zeros((self.states_num, self.actions_num))
        C = np.zeros((self.states_num, self.actions_num))
        Pi = self.get_greedy_policy()

        step, accuracies, mean_accuracies = 0, [], []
        num_of_discarded_trajectories = 0

        while step < self.cfg.num_episodes:

            if step % 100 == 0:
                acc = np.mean(np.asarray([np.argmax(Pi[s]) for s in range(self.states_num)]) ==
                              np.asarray(self.optimal_policy))
                accuracies.append(acc)
                mean_acc = np.mean(accuracies[:-10]) if len(accuracies) > 10 else np.mean(accuracies)
                print(f'step: {step}/{self.cfg.num_episodes}, accuracy: {mean_acc}')
                mean_accuracies.append(mean_acc)

            b_policy = self.get_random_policy()
            episode = self.generate_episode(b_policy)

            G, W = 0, 1

            for t in range(len(episode))[::-1]:
                state, action, reward = episode[t]
                state, action = int(state), int(action)

                W = W * 1. / b_policy[state][action]
                G = self.cfg.discount_factor * G + reward
                C[state][action] += W
                Q[state][action] += (W / C[state][action]) * (G - Q[state][action])

                if not with_correction:
                    Pi[state] = np.zeros(self.actions_num)
                    Pi[state][np.argmax(Q[state])] = 1

                if action != np.argmax(Pi[state]):
                    num_of_discarded_trajectories += 1
                    break

            if with_correction:
                for state in range(self.states_num):
                    Pi[state] = np.zeros(self.actions_num)
                    Pi[state][np.argmax(Q[state])] = 1

            step += 1

        return mean_accuracies, num_of_discarded_trajectories

    def off_policy_evaluation(self, use_weighted_importance_sampling):
        """
        Runs off policy evaluation algorithm.
        """
        Q = np.zeros((self.states_num, self.actions_num))
        C = np.zeros((self.states_num, self.actions_num))
        Pi = np.ones((self.states_num, self.actions_num)) / self.actions_num

        step, accuracies, mean_accuracies = 0, [], []
        all_nums_of_discarded_trajectories = []
        num_of_discarded_trajectories = 0

        while step < self.cfg.num_episodes:

            if step % 100 == 0:
                print(f'step: {step}/{self.cfg.num_episodes}')

            b_policy = self.get_random_policy()
            episode = self.generate_episode(b_policy)

            G, W = 0, 1

            for t in range(len(episode))[::-1]:
                state, action, reward = episode[t]
                state, action = int(state), int(action)

                W = W * (Pi[state][action] / b_policy[state][action])

                if W == 0:
                    num_of_discarded_trajectories += 1
                    break

                G = self.cfg.discount_factor * G + reward
                value_to_add = W if use_weighted_importance_sampling else 1
                C[state][action] += value_to_add
                Q[state][action] += (W / C[state][action]) * (G - Q[state][action])

            step += 1

        V = np.mean(Q, 1)
        return V, num_of_discarded_trajectories

    def run(self):
        """
        Runs off policy control/evaluation algorithms.
        """
        self.get_optimal_policy()

        # print('\nGetting V with direct solution...')
        v_direct = self.get_direct_solution()

        print('\nOff policy evaluation, ordinary importance sampling...\n')
        v_off_policy_eval_ordinary_sampling, _ = self.off_policy_evaluation(use_weighted_importance_sampling=False)

        print('\nOff policy evaluation, weighted importance sampling...\n')
        v_off_policy_eval_weighted_sampling, _ = self.off_policy_evaluation(use_weighted_importance_sampling=True)

        ordinary_sampling_bias, ordinary_sampling_var = self.get_bias_and_variance(v_direct,
                                                                                   v_off_policy_eval_ordinary_sampling)
        weighted_sampling_bias, weighted_sampling_var = self.get_bias_and_variance(v_direct,
                                                                                   v_off_policy_eval_weighted_sampling)

        print(f'\nBias (ordinary importance sampling - direct solution): {ordinary_sampling_bias}'
              f'\nVar (ordinary importance sampling - direct solution): {ordinary_sampling_var}\n'
              f'\nBias (weighted importance sampling - direct solution): {weighted_sampling_bias}'
              f'\nVar (weighted importance sampling - direct solution): {weighted_sampling_var}\n')

        print('\nOff policy control, without correction...\n')
        acc_off_policy_control_no_correction, discarded_trajectories_num_no_correction = \
            self.off_policy_control(with_correction=False)

        print('\nOff policy control, with correction...\n')
        acc_off_policy_control_with_correction, discarded_trajectories_num_with_correction = \
            self.off_policy_control(with_correction=True)

        print(f'\nDiscarded trajectories part (with correction): {discarded_trajectories_num_with_correction}'
              f'\nDiscarded trajectories part (no correction): {discarded_trajectories_num_no_correction}\n')

        off_policy_eval_results = (ordinary_sampling_bias, ordinary_sampling_var, weighted_sampling_bias,
                                   weighted_sampling_var)
        off_policy_control_results = (acc_off_policy_control_no_correction, acc_off_policy_control_with_correction,
                                      discarded_trajectories_num_no_correction,
                                      discarded_trajectories_num_with_correction)
        return off_policy_eval_results, off_policy_control_results
