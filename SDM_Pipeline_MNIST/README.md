### Basic forward diffusion

**Let's start with forward diffusion.** In the simplest case, the relevant diffusion equation is


x(t + \Delta t) = x(t) + \sigma(t) \sqrt{\Delta t} \ r


where $\sigma(t) > 0$ is the 'noise strength', $\Delta t$ is the step size, and $r \sim \mathcal{N}(0, 1)$ is a standard normal random variable. In essence, we repeatedly add normally-distributed noise to our sample. Often, the noise strength $\sigma(t)$ is chosen to depend on time (i.e. it gets higher as $t$ gets larger).