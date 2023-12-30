# The basics of Gaussian Processes

**Def.** A stochastic process $y(x)$ is specified by giving the joint probability distribution for any finite set of values $y(x_1), \dots, y(x_N)$ in a consistent manner.  

**Def.** A Gaussian process is defined as a probability distribution over functions $y(x)$ such that the set of values $y(x)$ evaluated at an arbitrary set of points $x_1, dots, x_N$ jointly have a Gaussian distribution.

Suppose we have a training set $\{x_n, t_n\}_{n=1}^N$ and that we try to predict the target values using the following model:
$$
y(x;w) = w^T\phi(x)
$$
Where $\phi : \R^d \to \R^m $ is a non linear kernel and $w \in \R^m$ is the parameter vector of our model. If we define a probability distribution over $w$ as 
$$
p(w) = \mathcal{N}(w \mid 0, \alpha^{-1}I)
$$
where $\alpha$ is a hyperparameter. We are technically defining a probability distribution over our function $y(x;w)$ which relies on $w$. 

Let's do something unusual. Let $\bar y \in \R^N$ be a vector such that $\bar y_n = y(x_n)$. We can obtain this vector as $\bar y = \Phi w$ where $\Phi$ is the design matrix (i.e., a $N \times M$ matrix where the i-th row corresponds to $\phi(x_i)$). As $\bar y$ is a linear combination between our training data and a normally distributed random variable, its distribution is also Gaussian. The parameters can be obtained as follows:

* $\mathbb{E}[\bar y] = \Phi \mathbb{E}[w] = 0$
* $cov[\bar y] = \mathbb{E}[\bar y \bar y^T] = \Phi \mathbb{E}[ww^t] \Phi^T = \frac1\alpha \Phi\Phi^T = K$ (the Gram's matrix)

The cool thing to observe here is that the covariance matrix is entirely composed by kernel evaluations:
$$
K_{ij} = k(x_i, x_j) = \frac1\alpha \phi(x_i)^T \phi(x_j)
$$
In this case the kernel is very simple, but we can replace it with another arbitrary valid kernel, this will allow us to build more complex models and it is a great feature of Gaussian Processes.

## Gaussian processes for regression

In our model, we are going to assume that every target variable has an independent noise term $\epsilon_n \sim \mathcal{N}(0, \beta^{-1})$ separating our prediction from the observed value:
$$
t_n = y_n + \epsilon_n
$$
Therefore the conditional probability distribution of the target variable is:
$$
p(t_n \mid y_n) = \mathcal{N}(t_n \mid y_n, \beta^{-1})
$$
Similarly to what we have done in the past section, let's define $\bar t \in \R^N$ such that $\bar t_n = t_n$.  Thanks to the assumption of independent noise, the previous equation can be generalized as:
$$
p(\bar t \mid \bar y) = \mathcal{N}(\bar t \mid \bar y, \beta^{-1}I)
$$
 Turns out this can be marginalized easily:
$$
p(\bar t) = \int p(\bar t \mid \bar y) d\bar y = \mathcal{N}(\bar t \mid 0, C)
$$
where $C$ is an $N \times N$ matrix such that
$$
C_{ij} = \begin{cases}
k(x_i, x_j) + \beta^{-1} & i=j \\
k(x_i, x_j) & i \ne j
\end{cases}
$$
Now suppose we want to infer $t_{N+1}$ from a test point $x_{N+1}$. Build $\bar t_{N+1}$ by extending $\bar t$ with another component $t_{N+1}$, then the distribution will be:
$$
p(\bar t _{N+1}) = \mathcal{N}(\bar t_{N+1} \mid 0, C_{N+1})
$$
Where $C_{N+1}$ will be a $(N+1) \times (N+1)$ matrix build like this:
$$
C_{N+1} = \begin{bmatrix}
C & k \\
k^T & c
\end{bmatrix}
$$
We already know $C$. $k \in \R^N$ is a vector such that $k_i = k(x_{N+1}, x_i)$ and finally $c = k(x_{N+1}, x_i) + \beta^{-1}$. We know the joint distribution $p(\bar t _{N+1}) = p(t_1, \dots, t_N, t_{N+1})$, but we are interested in predicting $t_{N+1}$, so we are interested in the posterior, which is another Gaussian distribution:
$$
p(t_{N+1} \mid t_1, \dots, t_N) = \mathcal{N}(t_{N+1}, \mu(x_n), \sigma^2(x_n))
$$
Results 2.81 and 2.82 from the PRML show how to compute the parameters:
$$
\mu(x_n) = k^T C \bar t \hspace{1cm}
\sigma^2(x_n) = c - k^T C^{-1} k
$$
Both quantities depend on $x_n$ as $k$ and $c$ terms also depend on it. This is it (for now). We can use an arbitrary valid kernel $k(\cdot, \cdot)$ as long as the resulting matrix $C$ is invertible.

