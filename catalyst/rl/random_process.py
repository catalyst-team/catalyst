import numpy as np


class RandomProcess(object):
    def __init__(self, sigma=0):
        pass

    def reset_states(self):
        pass

    def sample(self):
        return 0

    @property
    def current_sigma(self):
        return 0


class AnnealedGaussianProcess(RandomProcess):
    def __init__(self, mu, sigma, sigma_min, n_steps_annealing):
        super().__init__(sigma)
        self.mu = mu
        self.sigma = sigma
        self.n_steps = 0

        if sigma_min is not None:
            self.m = -float(sigma - sigma_min) / float(n_steps_annealing)
            self.c = sigma
            self.sigma_min = sigma_min
        else:
            self.m = 0.
            self.c = sigma
            self.sigma_min = sigma

    @property
    def current_sigma(self):
        sigma = max(self.sigma_min, self.m * float(self.n_steps) + self.c)
        return sigma


class GaussianWhiteNoiseProcess(AnnealedGaussianProcess):
    def __init__(
        self, mu=0., sigma=1., sigma_min=None, n_steps_annealing=1000, size=1
    ):
        super().__init__(
            mu=mu,
            sigma=sigma,
            sigma_min=sigma_min,
            n_steps_annealing=n_steps_annealing
        )
        self.size = size

    def sample(self):
        sample = np.random.normal(self.mu, self.current_sigma, self.size)
        self.n_steps += 1
        return sample


class OrnsteinUhlenbeckProcess(AnnealedGaussianProcess):
    def __init__(
        self,
        theta,
        mu=0.,
        sigma=1.,
        dt=1e-2,
        x0=None,
        size=1,
        sigma_min=None,
        n_steps_annealing=1000
    ):
        super().__init__(
            mu=mu,
            sigma=sigma,
            sigma_min=sigma_min,
            n_steps_annealing=n_steps_annealing
        )
        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.reset_states()

    def sample(self):
        theta = self.theta * (self.mu - self.x_prev)
        sqrt = np.sqrt(self.dt) * np.random.normal(size=self.size)
        x = self.x_prev + theta * self.dt + self.current_sigma * sqrt

        self.x_prev = x
        self.n_steps += 1
        return x

    def reset_states(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)
