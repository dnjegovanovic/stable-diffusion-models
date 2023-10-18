import unittest

import matplotlib.pyplot as plt

from SDM_Pipeline_MNIST.tools.forward_reverse_diffusion import *


class TestForwardReverseDiffusionFun(unittest.TestCase):
    def test_forward_diffusion(self):
        n_steps = 100
        t0 = 0
        dt = 0.1
        noise_strength_fn = noise_strength_constant
        x0 = 0

        num_tries = 5
        for _ in range(num_tries):
            x, t = forward_diffusion_1d(x0, noise_strength_fn, t0, n_steps, dt)

            plt.plot(t, x)
            plt.xlabel("time", fontsize=20)
            plt.ylabel("$x$", fontsize=20)

        plt.savefig("./playground_imgs/SDM_Pipeline_MNIST_imgs/forward_diffusin.jpg")

    def test_reverse_diffusion_1d(self):
        nsteps = 100
        dt = 0.1
        noise_strength_fn = noise_strength_constant
        score_fn = score_simple
        x0 = 0
        T = 11

        num_tries = 5
        for i in range(num_tries):
            x0 = np.random.normal(
                loc=0, scale=T
            )  # draw from the noise distribution, which is diffusion for time T w noise strength 1
            x, t = reverse_diffusion_1d(x0, noise_strength_fn, score_fn, T, nsteps, dt)

            plt.plot(t, x)
            plt.xlabel("time", fontsize=20)
            plt.ylabel("$x$", fontsize=20)

        plt.savefig("./playground_imgs/SDM_Pipeline_MNIST_imgs/reverse_diffusin.jpg")
