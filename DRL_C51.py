import numpy as np
import func
import math
import os
import time
import datetime
import tqdm

np.random.seed(808)

class Xtates():
    def __init__(self, my_map, num_atoms):
        self.state_size = my_map.M.shape[0]
        self.action_size = my_map.M.shape[1]
        self.num_atoms = num_atoms

        self.xtates = [{'state': i + 1, 'actions': [], 'state_prime': [], 'distributions': []} for i in
                       range(self.state_size)]
        self.xtates = np.array(self.xtates)

        M_T = my_map.M.transpose()

        for i in range(self.action_size):
            self.xtates[int(np.argwhere(M_T[i] == 1))]['actions'].append(i + 1)
            a = np.ones(self.num_atoms) / self.num_atoms  # np.random.rand(self.num_atoms)
            self.xtates[int(np.argwhere(M_T[i] == 1))]['distributions'].append(a)  # (a/np.sum(a))
            self.xtates[int(np.argwhere(M_T[i] == 1))]['state_prime'].append(int(np.argwhere(M_T[i] == -1)) + 1)

    def load_xtates(self, file_name):
        self.xtates = np.load(file_name, allow_pickle=True)
        self.xtates = self.xtates.tolist()

    def save_xtates(self, file_name):
        np.save(file_name, self.xtates)


class DRL_Agent():
    def __init__(self, X, my_map, termination):
        self.map = my_map
        self.xtates = X.xtates.copy()
        self.xtates_target = self.xtates.copy()
        self.termination = termination
        self.state_size = X.state_size
        self.action_size = X.action_size
        self.num_atoms = X.num_atoms
        # hyper parameters
        self.K = 5
        self.lr_rate = 0.1
        self.dynamic_lr = 0
        self.gamma = 1
        self.epsilon = 1.0
        self.initial_epsilon = 1.0
        self.final_epsilon = 0.1
        self.n_update_target = 100
        self.explorer_ratio = 0.9

        # Initialize Atoms
        self.v_max = 0
        self.v_min = -200
        self.delta_z = (self.v_max - self.v_min) / float(X.num_atoms - 1)
        self.z = np.array([self.v_min + i * self.delta_z for i in range(X.num_atoms)])

    def update_V(self, v_min, v_max):
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (self.v_max - self.v_min) / float(self.num_atoms - 1)
        self.z = np.array([self.v_min + i * self.delta_z for i in range(self.num_atoms)])

    def load_xtates(self, file_name):
        self.xtates = np.load(file_name, allow_pickle=True)

    def save_xtates(self, file_name):
        print("save xtates to {}".format(file_name))
        np.save(file_name, self.xtates)

    # \epsilon-greedy
    # (1-epsilon) probability to select an action from max(Q(s,a))
    def get_action(self, xtates, state, parameter=0.1, obj="LET"):
        if np.random.rand() <= self.epsilon:
            # get a random action
            action_idx = np.random.randint(0, len(self.xtates[state - 1]['actions']))
            action = self.xtates[state - 1]['actions'][action_idx]
            state_prime = self.xtates[state - 1]['state_prime'][action_idx]
        else:
            action_idx, action, state_prime = self.get_optimal_action(xtates, state, parameter=parameter, obj=obj)

        while len(self.xtates[state_prime-1]["actions"]) == 0 and state_prime != self.termination:
            action_idx = np.random.randint(0, len(self.xtates[state - 1]['actions']))
            action = self.xtates[state - 1]['actions'][action_idx]
            state_prime = self.xtates[state - 1]['state_prime'][action_idx]

        return action_idx, action, state_prime

    # a* =  argmax_(a) Q(s, a)
    # Q(s, a) \sigma zi*pi
    # select an action from max(Q(s,a))
    def get_optimal_action(self, xtates, state, parameter=0.1, obj="LET"):
        distributions = np.array(xtates[state - 1]['distributions'])
        distributions_concat = np.vstack(distributions)

        if obj == "LET":
            q = np.sum(np.multiply(distributions_concat, np.array(self.z)), axis=1)
            action_idx = np.argmax(q)
            optimal_action = xtates[state - 1]['actions'][action_idx]
            state_prime = xtates[state - 1]['state_prime'][action_idx]
        elif obj == "dij":
            action_idx, optimal_action, state_prime = self.get_dijkstra_action(state)
        elif obj == "SOTA":
            T = parameter * self.find_shortest_path(state)[0]
            tmp = int(T // self.delta_z)
            distributions_concat[:, 0: self.num_atoms-tmp-1] = 0
            q = np.sum(distributions_concat, axis=1)
            action_idx = np.argmax(q)
            optimal_action = xtates[state - 1]['actions'][action_idx]
            state_prime = xtates[state - 1]['state_prime'][action_idx]

        elif obj == "SOTA_T":
            T = parameter
            tmp = int(T // self.delta_z)
            distributions_concat[:, 0: self.num_atoms-tmp-1] = 0
            q = np.sum(distributions_concat, axis=1)
            action_idx = np.argmax(q)
            optimal_action = xtates[state - 1]['actions'][action_idx]
            state_prime = xtates[state - 1]['state_prime'][action_idx]

        elif obj == "mean-std":
            zeta = parameter
            ave = np.sum(np.multiply(distributions_concat, np.array(self.z)), axis=1)
            tmp = np.square(np.tile(self.z, distributions_concat.shape[0]).reshape(distributions_concat.shape[0], -1) - ave.reshape(-1, 1))
            std2 = np.sum(np.multiply(distributions_concat, tmp), axis=1)
            q = -ave + zeta * np.sqrt(std2)
            action_idx = np.argmin(q)
            optimal_action = xtates[state - 1]['actions'][action_idx]
            state_prime = xtates[state - 1]['state_prime'][action_idx]
        elif obj == "alpha":
            alpha = parameter
            prob = np.zeros((distributions_concat.shape[0], 1))
            action_idx = None
            for i in range(self.num_atoms-1, -1, -1):
                prob += distributions[:, i].reshape(-1, 1)
                if np.argwhere(prob >= alpha).size:
                    action_idx = np.argwhere(prob >= alpha)[0][0]
                    break
            if action_idx == None:
                action_idx = np.argmax(prob)
            optimal_action = xtates[state - 1]['actions'][action_idx]
            state_prime = xtates[state - 1]['state_prime'][action_idx]
        elif obj == "MED":
            alpha = parameter
            q = np.sum(np.multiply(distributions_concat, np.exp(-alpha*np.array(self.z))), axis=1)
            action_idx = np.argmin(q)
            optimal_action = xtates[state - 1]['actions'][action_idx]
            state_prime = xtates[state - 1]['state_prime'][action_idx]

        return action_idx, optimal_action, state_prime

    # select an action based on dijkstra
    # can lead distributions to convergence faster
    def get_dijkstra_action(self, state):
        flag, path, _ = func.dijkstra(self.map.G, state - 1, self.termination - 1)
        assert flag != -1
        action = path[0] + 1
        action_idx = self.xtates[state - 1]['actions'].index(action)
        state_prime = self.xtates[state - 1]['state_prime'][action_idx]
        return action_idx, action, state_prime

    # if we have no prior-experience, try use a more reliable method(dijkstra) to update distribututions
    # lead distributions to convergence faster
    def C51_with_dijkstra(self, state, action_idx, reward, state_prime):
        m_prob = np.zeros(self.num_atoms)
        if state_prime == self.termination:
            for r in reward:
                # Distribution collapses to a single point
                Tz = min(self.v_max, max(self.v_min, r))
                bj = (Tz - self.v_min) / self.delta_z
                m_l, m_u = math.floor(bj), math.ceil(bj)
                m_prob[int(m_l)] += (m_u - bj) / self.K
                m_prob[int(m_u)] += (bj - m_l) / self.K
            self.xtates[state - 1]['distributions'][action_idx] = self.xtates[state - 1]['distributions'][
                                                                      action_idx] * (
                                                                              1 - self.lr_rate) + self.lr_rate * m_prob
        else:
            action_star_idx, action_star, _ = self.get_dijkstra_action(state_prime)
            for r in reward:
                for j in range(self.num_atoms):
                    Tz = min(self.v_max, max(self.v_min, r + self.gamma * self.z[j]))
                    bj = (Tz - self.v_min) / self.delta_z
                    m_l, m_u = math.floor(bj), math.ceil(bj)
                    m_prob[int(m_l)] += self.xtates[state_prime - 1]['distributions'][action_star_idx][j] * (
                                m_u - bj) / self.K
                    m_prob[int(m_u)] += self.xtates[state_prime - 1]['distributions'][action_star_idx][j] * (
                                bj - m_l) / self.K
            self.xtates[state - 1]['distributions'][action_idx] = self.xtates[state - 1]['distributions'][
                                                                      action_idx] * (
                                                                              1 - self.lr_rate) + self.lr_rate * m_prob

    # on-policy to update dsitrbutions
    def C51_on_policy(self, state, action_idx, reward, state_prime, parameter=None, obj="LET"):
        m_prob = np.zeros(self.num_atoms)
        if state_prime == self.termination:
            for r in reward:
                # Distribution collapses to a single point
                Tz = min(self.v_max, max(self.v_min, r))
                bj = (Tz - self.v_min) / self.delta_z
                m_l, m_u = math.floor(bj), math.ceil(bj)
                m_prob[int(m_l)] += (m_u - bj) / self.K
                m_prob[int(m_u)] += (bj - m_l) / self.K
            self.xtates[state - 1]['distributions'][action_idx] = self.xtates[state - 1]['distributions'][
                                                                      action_idx] * (
                                                                              1 - self.lr_rate) + self.lr_rate * m_prob
        else:
            action_star_idx, action_star, _ = self.get_optimal_action(self.xtates, state_prime, parameter=parameter, obj=obj)
            for r in reward:
                for j in range(self.num_atoms):
                    Tz = min(self.v_max, max(self.v_min, r + self.gamma * self.z[j]))
                    bj = (Tz - self.v_min) / self.delta_z
                    m_l, m_u = math.floor(bj), math.ceil(bj)
                    m_prob[int(m_l)] += self.xtates[state_prime - 1]['distributions'][action_star_idx][j] * (
                                m_u - bj) / self.K
                    m_prob[int(m_u)] += self.xtates[state_prime - 1]['distributions'][action_star_idx][j] * (
                                bj - m_l) / self.K
            self.xtates[state - 1]['distributions'][action_idx] = self.xtates[state - 1]['distributions'][
                                                                      action_idx] * (
                                                                              1 - self.lr_rate) + self.lr_rate * m_prob

    # off-policy with Important Sample to update dsitrbutions
    def C51_IS(self, step, state, action_idx, reward, state_prime, parameter=None, obj="LET"):
        self.xtates_target = self.xtates.copy()
        m_prob = np.zeros(self.num_atoms)
        if (state_prime == self.termination):
            for r in reward:
                # Distribution collapses to a single point
                Tz = min(self.v_max, max(self.v_min, r))
                bj = (Tz - self.v_min) / self.delta_z
                m_l, m_u = math.floor(bj), math.ceil(bj)
                m_prob[int(m_l)] += (m_u - bj) / self.K
                m_prob[int(m_u)] += (bj - m_l) / self.K
            self.xtates[state - 1]['distributions'][action_idx] = self.xtates[state - 1]['distributions'][
                                                                      action_idx] * (
                                                                              1 - self.lr_rate) + self.lr_rate * m_prob
        else:
            action_star_idx, action_star, _ = self.get_optimal_action(self.xtates_target, state_prime, parameter=parameter, obj=obj)
            for r in reward:
                for j in range(self.num_atoms):
                    Tz = min(self.v_max, max(self.v_min, r + self.gamma * self.z[j]))
                    bj = (Tz - self.v_min) / self.delta_z
                    m_l, m_u = math.floor(bj), math.ceil(bj)
                    m_prob[int(m_l)] += self.xtates_target[state_prime - 1]['distributions'][action_star_idx][j] * (
                                m_u - bj) / self.K
                    m_prob[int(m_u)] += self.xtates_target[state_prime - 1]['distributions'][action_star_idx][j] * (
                                bj - m_l) / self.K
            self.xtates[state - 1]['distributions'][action_idx] = self.xtates[state - 1]['distributions'][
                                                                      action_idx] * (
                                                                              1 - self.lr_rate) + self.lr_rate * m_prob
        if (step % self.n_update_target == 0):
            self.xtates_target = self.xtates.copy()

    def find_shortest_path(self, start, is_print=False):
        cost, path, _ = func.dijkstra(self.map.G, start - 1, self.termination - 1)
        # assert cost != -1
        state = start
        state_path = []
        state_path.append(state)
        for action in path:
            action_idx = self.xtates[state - 1]['actions'].index(action + 1)
            state = self.xtates[state - 1]['state_prime'][action_idx]
            state_path.append(state)
        path = np.array(path) + 1

        if is_print:
            print('state_path:' + str(state_path))
            print('path: ' + str(path))
            print('toatal_cost = ' + str(cost))
        return cost, path, state_path

    def train_dijkstra(self, num_iterations, file_name=None, save_path=None):
        self.epsilon = self.initial_epsilon
        if (save_path != None):
            time_mark = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
            save_dir_path = save_path + '/' + time_mark
            os.mkdir(save_dir_path)
        if (file_name != None):
            self.load_xtates(file_name)
        for episode in tqdm.trange(num_iterations):
            if self.dynamic_lr:
                self.lr_rate = 1 / np.sqrt(episode + 1)
            state = np.random.randint(1, self.state_size+1)
            while len(self.xtates[state-1]["actions"]) == 0:
                state = np.random.randint(1, self.state_size+1)
            total_reward = 0
            print('------------{}th episode is starting.----------'.format(episode))
            state_path = []
            path = []
            while True:
                if (state == self.termination):
                    state_path.append(state)
                    break
                else:
                    state_path.append(state)
                    action_idx, action, state_prime = self.get_action(self.xtates, state, obj="LET")
                    path.append(action)
                    sigma2 = self.map.sigma[action - 1][action - 1]
                    mu = self.map.mu[action - 1]
                    cov2_log = np.log(sigma2 / mu ** 2 + 1)
                    mu_log = np.log(mu ** 2 / (np.sqrt(sigma2 + mu ** 2)))
                    reward = -np.random.lognormal(mu_log, np.sqrt(cov2_log), self.K)
                    # reward = -np.random.normal(self.map.mu[action - 1], self.map.sigma[action - 1][action - 1], self.K)
                    total_reward += np.sum(reward) / self.K
                    self.C51_with_dijkstra(state, action_idx, reward, state_prime)
                    state = state_prime
                if total_reward <= self.v_min*4:
                    print("!!!!!!!!!!!!!!!!!explorer failure!!!!!!!!!!!!!!!!!!!")
                    break

            if (((episode + 1) % 10000 == 0) and (save_path != None)):
                self.save_xtates(save_dir_path + '/V200_{}W'.format(episode // 10000 + 1) + '.npy')

            print('toal_reward:{}'.format(total_reward))
            print('state_path: ' + str(state_path))
            print('path: ' + str(path))
            if (self.epsilon >= self.final_epsilon):
                self.epsilon -= (self.initial_epsilon - self.final_epsilon) / (num_iterations * self.explorer_ratio)

    def train_on_policy(self, num_iterations, file_name=None, save_path=None, parameter=0.1, obj="LET"):
        self.epsilon = self.initial_epsilon
        if (save_path != None):
            time_mark = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
            save_dir_path = save_path + '/' + time_mark
            os.mkdir(save_dir_path)
        if (file_name != None):
            self.load_xtates(file_name)
        for episode in range(num_iterations):
            if self.dynamic_lr:
                self.lr_rate = 1 / np.sqrt(episode + 1)
            state = np.random.randint(1, self.state_size+1)
            total_reward = 0
            print('------------{}th episode is starting.----------'.format(episode))
            state_path = []
            path = []
            while True:
                if (state == self.termination):
                    state_path.append(state)
                    break
                else:
                    state_path.append(state)
                    action_idx, action, state_prime = self.get_action(self.xtates, state, parameter=parameter, obj=obj)
                    path.append(action)
                    sigma2 = self.map.sigma[action - 1][action - 1]
                    mu = self.map.mu[action - 1]
                    cov2_log = np.log(sigma2 / mu ** 2 + 1)
                    mu_log = np.log(mu ** 2 / (np.sqrt(sigma2 + mu ** 2)))
                    reward = -np.random.lognormal(mu_log, np.sqrt(cov2_log), self.K)
                    # reward = -np.random.normal(self.map.mu[action - 1], self.map.sigma[action - 1][action - 1], self.K)
                    total_reward += np.sum(reward) / self.K
                    self.C51_on_policy(state, action_idx, reward, state_prime, parameter=parameter, obj=obj)
                    state = state_prime

            if (((episode + 1) % 10000 == 0) and (save_path != None)):
                self.save_xtates(save_dir_path + '/V200_{}W'.format(episode // 10000 + 1) + '.npy')

            print('toal_reward:{}'.format(total_reward))
            print('state_path: ' + str(state_path))
            print('path: ' + str(path))
            if (self.epsilon >= self.final_epsilon):
                self.epsilon -= (self.initial_epsilon - self.final_epsilon) / (num_iterations * self.explorer_ratio)

    # off-policy with importance sample
    def train_IS(self, num_iterations, file_name=None, save_path=None, parameter=0.1, obj="LET"):
        self.epsilon = self.initial_epsilon
        if (save_path != None):
            time_mark = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
            save_dir_path = save_path + '/' + time_mark
            os.mkdir(save_dir_path)
        if (file_name != None):
            self.load_xtates(file_name)
            self.xtates_target = self.xtates.copy()
        step = 0
        for episode in tqdm.trange(num_iterations):
            if self.dynamic_lr:
                self.lr_rate = 1 / np.sqrt(episode + 1)
            state = np.random.randint(1, self.state_size+1)
            while len(self.xtates[state - 1]["actions"]) == 0:
                state = np.random.randint(1, self.state_size+1)
            total_reward = 0
            print('------------{}th episode is starting.----------'.format(episode))
            state_path = []
            path = []
            while True:
                if (state == self.termination):
                    state_path.append(state)
                    break
                else:
                    state_path.append(state)
                    action_idx, action, state_prime = self.get_action(self.xtates, state, parameter=parameter, obj=obj)
                    step += 1
                    path.append(action)
                    sigma2 = self.map.sigma[action - 1][action - 1]
                    mu = self.map.mu[action - 1]
                    cov2_log = np.log(sigma2 / mu ** 2 + 1)
                    mu_log = np.log(mu ** 2 / (np.sqrt(sigma2 + mu ** 2)))
                    reward = -np.random.lognormal(mu_log, np.sqrt(cov2_log), self.K)
                    # reward = -np.random.normal(self.map.mu[action - 1], self.map.sigma[action - 1][action - 1], self.K)
                    total_reward += np.sum(reward) / self.K
                    self.C51_IS(step, state, action_idx, reward, state_prime, parameter=parameter, obj=obj)
                    state = state_prime
                if total_reward <= self.v_min*4:
                    print("!!!!!!!!!!!!!!!!!explorer failure!!!!!!!!!!!!!!!!!!!")
                    break

            if (((episode + 1) % 10000 == 0) and (save_path != None)):
                self.save_xtates(save_dir_path + '/V200_{}W'.format(episode // 10000 + 1) + '.npy')

            print('toal_reward:{}'.format(total_reward))
            print('state_path: ' + str(state_path))
            print('path: ' + str(path))
            if (self.epsilon >= self.final_epsilon):
                self.epsilon -= (self.initial_epsilon - self.final_epsilon) / (num_iterations * self.explorer_ratio)

    def train_Linear_test(self, num_iterations):
        # self.epsilon = self.initial_epsilon
        step = 0
        for episode in range(num_iterations):
            if self.dynamic_lr:
                self.lr_rate = 1 / np.sqrt(episode + 1)
            state = np.random.randint(1, self.state_size+1)
            total_reward = 0
            print('------------{}th episode is starting.----------'.format(episode))
            state_path = []
            path = []
            while True:
                if (state == self.termination):
                    state_path.append(state)
                    break
                else:
                    state_path.append(state)
                    action_idx, action, state_prime = self.get_action(self.xtates, state, parameter=0, obj="LET")
                    step += 1
                    path.append(action)
                    if action == 1:
                        reward = -np.random.normal(10, 1, self.K)
                    elif action == 2:
                        sigma2 = self.map.sigma[action - 1][action - 1]
                        mu = self.map.mu[action - 1]
                        cov2_log = np.log(sigma2 / mu ** 2 + 1)
                        mu_log = np.log(mu ** 2 / (np.sqrt(sigma2 + mu ** 2)))
                        reward = -np.random.lognormal(mu_log, np.sqrt(cov2_log), self.K)
                    elif action == 3:
                        reward = -11 * np.random.weibull(11, self.K)
                    elif action == 4:
                        sigma2 = self.map.sigma[action - 1][action - 1]
                        mu = self.map.mu[action - 1]
                        cov2_gamma = mu ** 2 / sigma2
                        mu_gamma = sigma2 / mu
                        reward = -np.random.gamma(cov2_gamma, mu_gamma, self.K)

                    total_reward += np.sum(reward) / self.K
                    self.C51_IS(step, state, action_idx, reward, state_prime, parameter=0, obj="LET")
                    state = state_prime

            print('toal_reward:{}'.format(total_reward))
            print('state_path: ' + str(state_path))
            print('path: ' + str(path))
            if (self.epsilon >= self.final_epsilon):
                self.epsilon -= (self.initial_epsilon - self.final_epsilon) / (num_iterations * self.explorer_ratio)

    def find_path(self, start, parameter=0.1, obj = "LET", is_print = False):
        state = start
        total_reward = 0
        state_path = []
        path = []
        n_step = 0
        while True:
            if (state == self.termination):
                state_path.append(state)
                break
            else:
                action_idx, action, state_prime = self.get_optimal_action(self.xtates, state, parameter=parameter, obj=obj)
                if len(self.xtates[state_prime-1]["actions"]) == 0:
                    path = -1
                    print('---------------FATAL FAILURE------------------')
                    break
                    # action_idx = np.random.randint(0, len(self.xtates[state - 1]['actions']))
                    # action = self.xtates[state - 1]['actions'][action_idx]
                    # state_prime = self.xtates[state - 1]['state_prime'][action_idx]
                path.append(action)
                state_path.append(state)
                sigma2 = self.map.sigma[action - 1][action - 1]
                mu = self.map.mu[action - 1]
                cov2_log = np.log(sigma2 / mu ** 2 + 1)
                mu_log = np.log(mu ** 2 / (np.sqrt(sigma2 + mu ** 2)))
                reward = np.random.lognormal(mu_log, np.sqrt(cov2_log))
                total_reward += reward
                n_step += 1
                state = state_prime
            if (n_step >= self.state_size):
                path = -1
                print('-------------FATAL FAILURE------------------')
                break
        if is_print:
            print('state_path:' + str(state_path))
            print('path: ' + str(path))
            print('toatal_cost = ' + str(total_reward))
        return total_reward, path, state_path

    def find_SOTA_path(self, start, T, is_print = False):
        state = start
        total_reward = 0
        state_path = []
        path = []
        n_step = 0
        while True:
            if (state == self.termination):
                state_path.append(state)
                break
            else:
                action_idx, action, state_prime = self.get_optimal_action(self.xtates, state, parameter=T, obj="SOTA_T")

                path.append(action)
                state_path.append(state)
                sigma2 = self.map.sigma[action - 1][action - 1]
                mu = self.map.mu[action - 1]
                cov2_log = np.log(sigma2 / mu ** 2 + 1)
                mu_log = np.log(mu ** 2 / (np.sqrt(sigma2 + mu ** 2)))
                reward = np.random.lognormal(mu_log, np.sqrt(cov2_log))
                total_reward += reward
                T -= reward
                n_step += 1
                state = state_prime

            if len(self.xtates[state-1]["actions"]) == 0:
                path = -1
                break

            if (n_step >= self.state_size):
                path = -1
                break
        if is_print:
            print('state_path:' + str(state_path))
            print('path: ' + str(path))
            print('toatal_cost = ' + str(total_reward))
        return total_reward, path, state_path

    def get_RSP_result(self, start, parameter, obj):
        distributions = np.array(self.xtates[start - 1]['distributions'])
        distributions_concat = np.vstack(distributions)

        if obj == "SOTA":
            T = parameter * self.find_shortest_path(start)[0]
            prob = 0
            for i in range(100):
                # if self.find_path(start=start, parameter=parameter, is_print=False)[0] <= T:
                if self.find_SOTA_path(start=start, T=T, is_print=False)[0] <= T:
                    prob += 1 / 100
            return prob

        elif obj == "mean-std":
            zeta = parameter
            if self.find_path(start, zeta, obj="mean-std")[1] == -1:
                return -1
            link_path = np.array(self.find_path(start, zeta, obj="mean-std")[1])-1
            mean = np.sum(self.map.mu[link_path])
            std = np.sum(np.diag(self.map.sigma)[link_path])
            mean_std = mean + zeta * np.sqrt(std)
            return mean_std

        elif obj == "alpha":
            alpha = parameter
            prob = np.zeros((distributions_concat.shape[0], 1))
            for i in range(self.num_atoms - 1, -1, -1):
                prob += distributions[:, i].reshape(-1, 1)
                if np.argwhere(prob >= alpha).size:
                    min_time = -self.z[i]
                    break
            return min_time

        elif obj == "MED":
            alpha = parameter
            tmp = self.find_path(start, alpha, obj="MED")[1]
            if tmp == -1:
                # print("MED={} search failure.".format(alpha))
                return -1
            link_path = np.array(tmp)-1
            MED = 0
            for i in range(1000):
                cost = func.get_cost_from_path(self.map, link_path)
                MED += np.exp(alpha * cost)/1000
            return MED.item()


