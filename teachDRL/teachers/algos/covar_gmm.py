from sklearn.mixture import GaussianMixture as GMM
import numpy as np
from gym.spaces import Box

def proportional_choice(v, eps=0.):
    if np.sum(v) == 0 or np.random.rand() < eps:
        return np.random.randint(np.size(v))
    else:
        probas = np.array(v) / np.sum(v)
        return np.where(np.random.multinomial(1, probas) == 1)[0][0]

# Implementation of IGMM (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3893575/) + minor improvements
class CovarGMM():
    def __init__(self, mins, maxs, seed=None, params=dict()):
        self.seed = seed
        if not seed:
            self.seed = np.random.randint(42,424242)
        np.random.seed(self.seed)

        # Task space boundaries
        self.mins = np.array(mins)
        self.maxs = np.array(maxs)

        # Range of number of Gaussians to try when fitting the GMM
        self.potential_ks = np.arange(2, 11, 1) if "potential_ks" not in params else params["potential_ks"]
        # Ratio of randomly sampled tasks VS tasks sampling using GMM
        self.random_task_ratio = 0.2 if "random_task_ratio" not in params else params["random_task_ratio"]
        self.random_task_generator = Box(self.mins, self.maxs, dtype=np.float32)

        # Number of episodes between two fit of the GMM
        self.fit_rate = 250 if "fit_rate" not in params else params['fit_rate']
        self.nb_random = self.fit_rate  # Number of bootstrapping episodes

        # Original version do not use Absolute LP, only LP.
        self.absolute_lp = False if "absolute_lp" not in params else params['absolute_lp']

        self.tasks = []
        self.tasks_times_rewards = []
        self.all_times = np.arange(0, 1, 1/self.fit_rate)

        # boring book-keeping
        self.bk = {'weights': [], 'covariances': [], 'means': [], 'tasks_lps': [], 'episodes': []}

    def update(self, task, reward):
        # Compute time of task, relative to position in current batch of tasks
        current_time = self.all_times[len(self.tasks) % self.fit_rate]

        self.tasks.append(task)

        # Concatenate task with its corresponding time and reward
        self.tasks_times_rewards.append(np.array(task.tolist() + [current_time] + [reward]))

        if len(self.tasks) >= self.nb_random:  # If initial bootstrapping is done
            if (len(self.tasks) % self.fit_rate) == 0:  # Time to fit
                # 1 - Retrieve last <fit_rate> (task, time, reward) triplets
                cur_tasks_times_rewards = np.array(self.tasks_times_rewards[-self.fit_rate:])

                # 2 - Fit batch of GMMs with varying number of Gaussians
                potential_gmms = [GMM(n_components=k, covariance_type='full') for k in self.potential_ks]
                potential_gmms = [g.fit(cur_tasks_times_rewards) for g in potential_gmms]

                # 3 - Compute fitness and keep best GMM
                aics = [m.aic(cur_tasks_times_rewards) for m in potential_gmms]
                self.gmm = potential_gmms[np.argmin(aics)]

                # book-keeping
                self.bk['weights'].append(self.gmm.weights_.copy())
                self.bk['covariances'].append(self.gmm.covariances_.copy())
                self.bk['means'].append(self.gmm.means_.copy())
                self.bk['tasks_lps'] = self.tasks_times_rewards
                self.bk['episodes'].append(len(self.tasks))

    def sample_task(self):
        if (len(self.tasks) < self.nb_random) or (np.random.random() < self.random_task_ratio):
            # Random task sampling
            new_task = self.random_task_generator.sample()
        else:
            # Task sampling based on positive time-reward covariance

            # 1 - Retrieve positive time-reward covariance for each Gaussian
            self.times_rewards_covars = []
            for pos, covar, w in zip(self.gmm.means_, self.gmm.covariances_, self.gmm.weights_):
                if self.absolute_lp:
                    self.times_rewards_covars.append(np.abs(covar[-2,-1]))
                else:
                    self.times_rewards_covars.append(max(0, covar[-2, -1]))

            # 2 - Sample Gaussian according to its Learning Progress, defined as positive time-reward covariance
            idx = proportional_choice(self.times_rewards_covars, eps=0.0)

            # 3 - Sample task in Gaussian, without forgetting to remove time and reward dimension
            new_task = np.random.multivariate_normal(self.gmm.means_[idx], self.gmm.covariances_[idx])[:-2]
            new_task = np.clip(new_task, self.mins, self.maxs).astype(np.float32)

        return new_task

    def dump(self, dump_dict):
        dump_dict.update(self.bk)
        return dump_dict