# Kernel Density Estimators

Suppose that we want to estimate the value of $p(x)$, where $x \in \mathbb R^D$, and the PDF $p$ is totally unknown. We want to use a non-parametric estimation method. We suppose that $x$ lies in an Euclidean space. 

Let $R$ be a region containing the point $x$, the probability $P$ of falling into the region $x$ is defined by:

$$
P = \int_R p(x) dx
$$

Let $x_1, \dots, x_N$ be $N$ i.i.d. data points drawn from $p(x)$, then the probability that $K$ of $N$ data points fall into region $R$ is a binomial distribution:


$$
Bin(K \mid N, P) = \binom{N}{K} \mu^K (1 - \mu)^{N - K}
$$

Fromt the properties of the Binomial distribution, we have that:

1. $\mathbb E[K/N] = P$
2. $var[K/N] = P(1-P) / N$ becomes smaller with $N\to \infty$ 

We have to suppose (A) that the region $R$ is large enough such that the $N$ points are sufficient to get a sharply peaked binomial distribution. 

In that case, for $N \to \infty$ we have that $K/N \approx P$, and also:

$$
K \approx NP
$$

Now we suppose (B) that the region $R$ is sufficiently small such that the probability $p(x)$ is constant for $x \in R$. In this case, we have that:

$$
P \approx p(x)V
$$

where $V$ is the volume of the region $R$.

Observe that the two assumptions (A) and (B) are contradictory. Which is a limitation of the KDE methods. By assuming this, we can use the two results to derive:

$$
p(x) = \frac{K}{NV}
$$

The KDE methods are based on this result. They usually fix $V$ and then get $K$ from the $N$ observations available:

```python
def kde(points: List, region: Region) -> float:
    """
    Generic KDE estimator
    """
    K = 0
    N = len(points)
    for point in points:
        if point in region:
            K += 1
    return K / (N * region.volume)
```

## Parzen estimator

In the function above, we suppose that we can evaluate if the point lies inside the region, but this depends on the shape of the region we use. The Parzen estimator uses a hypercube centered at the point $x$ where we want to evaluate the PDF $p(x)$. 

We now introduce a kernel function $k(u)$, also called Parzen Window, defined as follows:

$$
k(u) = \begin{cases}
1 & \text{ if } |u_i| \le 1/2, \space i=1, \dots, D \\
0 & \text{ otherwise}
\end{cases}
$$

To know if a point $x_i$ lies inside the hypercube of side $h$ centered on $x$, we need to scale the point coordinates using this formula:

$$
k\left( \frac{x-x_i}{h} \right)
$$

In this way, we can compute $K$ by:

$$
K = \sum_{n=1}^N k\left( \frac{x-x_n}{h} \right)
$$

and since the volume of an hypercube of $D$ dimensions and of edge size $h$ is $V = h^D$, we can replace $K$ and $V$ in the $p(x)$ equation and get:

$$
p(x) = \frac1N \sum_{n=1}^N \frac1{h^D} \cdot k\left( \frac{x-x_n}{h} \right)
$$


```python
class ParzenWindow(Region):

    def __init__(self, h: float, origin: Point):
        self.h = h
        self.origin = origin
        self.volume = h ** len(origin)

    def __contains__(self, point):
        return self._k( (point - self.origin) / self.h )

    def _k(self, u):        
        return int(all([ u_n < 0.5 for u_n in u ]))
```

### Gaussian Kernel

The Parzen Estimator has discontinuities (boundaries of the cubes). A smoother kernel function is the Gaussian:

$$
p(x) = \frac1N \sum_{n=1}^N \frac{1}{\sqrt{2\pi}h^2} \exp \left\{
    - \frac{|| x - x_n ||^2}{2h^2}
\right\}
$$

Where $h$ now represents the standard deviation of the Gaussian components. 

In general, we can use every kernel function $k(u)$ that satisfies:

$$
k(u) \ge 0 \hspace{1cm} \int k(u)du = 1
$$

## Nearest-neighbour methods

A density distribution could have locations with smaller structures and location with bigger structures. KDE methods use the same $h$ parameter everywhere, and this may not be suitable for every location of the distribution.  

Nearest-neighbour methods address this problem: recall the general problem of non-parametric density estimation, let's fix K and use the data to find an appropiate value for V. 

We consider a small sphere centered on the point $x$ at which we wish to estimate the density $p(x)$, we allow the radius of the sphere to grow until it contains precisely $K$ data points. $V$ will be the volume of that sphere.

> The model produced by kNN is not a true density model since the integral over all spaces diverges.

> kNN can also be used as a classifier. An interesting property of kNN with $K=1$ is that in the limit $N \to \infty$, the error rate is never more than twice the minimum achievable error rate of an optimal classifier.