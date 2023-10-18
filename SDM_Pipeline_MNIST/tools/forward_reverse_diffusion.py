import numpy as np


def forward_diffusion_1d(x0, noise_strength_fn, t0, n_steps, dt):
    """x(t + Delta t) = x(t) + sigma(t) sqrt{Delta t} r

        where sigma(t) > 0 is the 'noise strength', Delta t is the step size,
        and r N(0, 1) is a standard normal random variable.
        In essence, we repeatedly add normally-distributed noise to our sample.
        Often, the noise strength sigma(t) is chosen to depend on time (i.e. it gets higher as $t$ gets larger).

    Args:
        x0 (scalar): initial sample value
        noise_stranght_fn (fun): function of time, outputs scalar noise strength
        t0 (_type_): initial time
        n_steps (_type_): number of diffusion steps
        dt (_type_): time step size
    """

    # Initialize trajectory
    x = np.zeros(n_steps + 1)
    x[0] = x0
    t = t0 + np.arange(n_steps + 1) * dt

    # Perform many Euler-Maruyama time steps
    for i in range(n_steps):
        noise_strength = noise_strength_fn(t[i])
        random_normal = np.random.randn()
        x[i + 1] = x[i] + np.sqrt(dt) * noise_strength * random_normal

    return x, t


# Example noise strength function: always equal to 1
def noise_strength_constant(t):
    return 1


def reverse_diffusion_1d(x0, noise_strength_fn, score_fn, T, n_steps, dt):
    """score function. If we know this function,
        we can reverse the forward diffusion and turn noise into what we started with.
        If our initial sample is always just one point at $x_0 = 0$,
        and the noise strength is constant, then the score function is exactly equal to
        s(x, t) = -(x - x_0)/sigma^2*t = -x/sigma^2*t

    Args:
        x0 (_type_): initial sample value
        noise_strength_fn (_type_): function of time, outputs scalar noise strength
        score_fn (_type_): score function
        T (_type_): final time
        n_steps (_type_): number of diffusion steps
        dt (_type_): time step size
    """

    # Initialize trajectory
    x = np.zeros(n_steps + 1)
    x[0] = x0
    t = np.arange(n_steps + 1) * dt

    # Perform many Euler-Maruyama time steps
    for i in range(n_steps):
        noise_strength = noise_strength_fn(T - t[i])
        score = score_fn(x[i], 0, noise_strength, T - t[i])
        random_normal = np.random.randn()
        x[i + 1] = (
            x[i]
            + (noise_strength**2) * score * dt
            + np.sqrt(dt) * noise_strength * random_normal
        )
    return x, t


def score_simple(x, x0, noise_strength, t):
    score = -(x - x0) / ((noise_strength**2) * t)
    return score
