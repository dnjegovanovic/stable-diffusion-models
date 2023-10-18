### Basic forward diffusion

**Let's start with forward diffusion.** In the simplest case, the relevant diffusion equation is


x(t + &Delta; t) = x(t) + &sigma;(t) &sqrt{&Delta; t} * r


where &sigma;(t) > 0$ is the 'noise strength', &Delta; t$ is the step size, and $r \sim \mathcal{N}(0, 1)$ is a standard normal random variable. In essence, we repeatedly add normally-distributed noise to our sample. Often, the noise strength $&sigma;(t) is chosen to depend on time (i.e. it gets higher as $t$ gets larger).

### Basic reverse diffusion

We can reverse this diffusion process by a similar-looking update rule:
is called the **score function**. If we know this function, we can reverse the forward diffusion and turn noise into what we started with.
If our initial sample is always just one point at $x_0 = 0$, and the noise strength is constant, then the score function is exactly equal to
s(x, t) = - (x - x_0)/&sigma;^2*t = - x / &sigma;^2*t
