# Probability distributions

This chapter will focus on the problem of density estimation, which consists in finding / estimating the probability distribution $p(x)$ from $N$ independent and identically distributed datapoints $x_1, x_2, \dots, x_N$ drawn from $p(x)$. There are two main ways of doing that: the first way is to use parametric density estimation, where you choose one known parametric distribution (e.g., Gaussian) and try to get the right parameters that fit the data. This method assumes that the parametric distribution we use it's suitable for the data, which is not always the case. Another way of doing that is using non-parametric density estimation techniques (e.g., histograms, nn, kernels).

## Bernoulli experiment

Suppose we have a data set $D = \{x_1, \dots, x_N \}$ of i.i.d. observed values of $x \sim Bern(x \mid \mu)$. We can estimate the $\mu$ parameter from the sample in a frequentist way, by maximizing the likelihood (or the log-likelihood): 

$$
\begin{split}
\ln p(D \mid \mu) &= \ln \prod_{n=1}^N p(x_i \mid \mu) \\
&= \sum_{n=1}^N \ln [ \mu^{x_n} (1-\mu)^{(1 - x_n)} ] \\
&= \sum_{n=1}^N [x_n\ln\mu + (1-x_n)\ln(1-\mu)] \\
\end{split} 
$$

To find $\mu$, let's set the log-likelihood derivative w.r.t. $\mu$ to 0:

$$
\begin{split}
\frac{\partial}{\partial \mu} \ln p(D \mid \mu) &= 0 \\
\sum_{n=1}^N \left(\frac{x_n}{\mu} + \frac{1 - x_n}{1-\mu}\right) &= 0 \\
\sum_{n=1}^N \frac{x_n - \mu}{\mu + \mu^2} &= 0 \\
\end{split} 
$$

Since $\frac{x_n - \mu}{\mu + \mu^2} = 0 \Leftrightarrow x_n = \mu$ then:

$$
\begin{split}
\sum_{n=1}^N x_n - \mu &= 0 \\
\sum_{n=1}^N x_n - \sum_{n=1}^N \mu &= 0 \\
\sum_{n=1}^N x_n &= N \mu \\
\frac1N \sum_{n=1}^N x_n &= \mu
\end{split}
$$

$\mu$ is estimated from the sample mean. In this case, the sample mean is an example of sufficient statistic for the model, i.e. calculating other statistics from the sample will not add more information than that.

## Binomial distribution

Sticking with the coin flips example, the binomial distribution models the probability of obtaining $m$ heads out of $N$ total coin flips:

$$
Bin(m \mid N, \mu) = \binom{N}{m} \mu^m (1 - \mu)^{N - m}
$$

Where $\binom{N}{m}$ represents all the possible ways of obtaining $m$ heads out of $N$ coin flips. The mean and variance of a binomial variabile can be estimated by knowning that for i.i.d events the mean of the sum is the sum of the mean, and the variance of the sum is the sum of the variances. Because $m = x_1 + \dots + x_N$ then:

$$
\mathbb E[m] = N \mu \hspace{1cm}
var[m] = N\mu(1-\mu)
$$

## Beta distribution

Please read the [`estimating_parameters_using_a_bayesian_approach`]([Title](https://github.com/LemuelPuglisi/pattern-recognition-and-machine-learning/blob/main/notebooks/ch2/estimating_parameters_using_a_bayesian_approach.ipynb)) notebook. Some quick notes here:

$$
\beta(\mu) = \frac{\Gamma(a+b)}{\Gamma(a) + \Gamma(b)} \mu^{(a-1)}(1-\mu)^{(b-1)}
$$

and 

$$
\mathbb E[\mu] = \frac{a}{a+b} \hspace{1cm}
var[\mu] = \frac{ab}{(a+b)^2(a+b+1)}
$$

## Multinomial variables

It's a generalization of the Bernoulli distribution where a random variable has $K$ possible values instead of being binary. We can represent the variable as a $K$-dimensional binary vector $x = \langle x_1, x_2, \dots, x_K \rangle$ where only one component can be asserted:

$$
\sum_{k=1}^K x_k = 1
$$

The probability of each component to be asserted is regulated by a probability vector $\mu = \langle \mu_1, \mu_2, \dots, \mu_K \rangle$, so that basically $x_k \sim Bern(\mu_k)$. Since the $\mu$ vector represents a probability distribution, then:

$$
\sum_{k=1}^K \mu_k = 1
$$

The multinomial distribution of $x$ is given by:

$$
p(x \mid \mu) = \prod_{k=1}^K \mu_k^{x_k}
$$

And the expected values is $\mathbb{E}[x \mid \mu] = \mu$. Let's consider a dataset of N independente observations, then the likelihood function is:

$$
p(D \mid \mu) = \prod_{n=1}^N \prod_{k=1}^K \mu_k^{x_{nk}} = 
\prod_{k=1}^K \mu_k^{(\sum_{n=1}^N x_{nk})} = 
\prod_{k=1}^K \mu_k^{m_k}
$$

where $m_k = \sum_{n=1}^N x_{nk}$. 

If we want to find $\mu$ from $D$ by maximizing the (log) likelihood, we have to constrain that to be a probability distribution and therefore we can use the Lagrangian multiplier $\lambda$

$$
\sum_{k=1}^K m_k \ln \mu_k + \lambda\left(\sum_{k=1}^K \mu_k - 1 \right)
$$

Setting the derivative w.r.t. $\mu_k$ to zero we get $\mu_k = -m_k/\lambda$. We can solve for the Lagrangian multiplier $\lambda$ by replacing this result in the equation and then we get that $\lambda = -N$ and $\mu_k^{ML} = m_k / N$. 

We can also consider the distribution of the quantities $m_1, \dots, m_k$ (**Multinomial distribution**) conditioned on the parameter $\mu$ and on the number $N$ of observations:

$$
Mult(m_1, dots, m_K \mid \mu, N) = \binom{N}{m_1m_2\dots m_K} \prod_{k=1}^K \mu_k^{m_k}
$$

where 

$$
\binom{N}{m_1m_2\dots m_K} = \frac{N!}{m_1!m_2!\dots m_K!} \hspace{1cm}
\sum_{k=1}^K m_k = N
$$

> [Short description of Lagrangian Multiplier utility taken from Quora](https://www.quora.com/What-do-people-use-Lagrange-Multipliers-for): 
> You are trying to maximize or minimize some function $f$ (distance to treasure), while keeping some other function $g$ fixed at a certain value $c$ (stay on the path). At this point, the gradient $\nabla f$ (the compass needle) must be parallel to the gradient $\nabla g$ (the arrows on the signs), but the two vectors will not generally have the same length. The test for whether or not they’re parallel is $\nabla f = \lambda \nabla g$, where $\lambda$ is whatever multiplier is needed to have them match; it will still only be able to be equal if they’re parallel (you can resize the compass needle however you want to make it match the sign arrow, but you have to be at a spot with the right direction).

## Dirichlet distribution

While the beta distribution is a prior of the Bernoulli parameter $\mu$, the Dirichlet distribution is a prior of the Multinomial probability vector $\bar \mu$. The definition is:

$$
Dir(\mu \mid \alpha) = \frac{\Gamma(\alpha_0)}{\Gamma(\alpha_1)\dots\Gamma(\alpha_K)} \prod_{k=1}^K \mu_k^{\alpha_k - 1}
$$

Where $\alpha_0 = \sum_{k=1}^K \alpha_k$. Since the $\bar\mu$ parameters are bounded to $\sum_k \mu_k = 1$, then the distribution is confined to a simplex in the $K-1$ space. 

By multiplying the likelihood function (which is the multinomial distribution) by the prior (which is a Dirichlet distribution) we get something that is proportional to the posterior $p(\mu \mid D, \alpha)$. Assuming a conjugate prior, the posterior has the same form and hence we can derive the normalization constant by comparison with the dirichlet distribution definition. The posterior is defined as:

$$
p(\mu \mid D, \alpha) = \frac{\Gamma(\alpha_0 + N)}{\Gamma(\alpha_1 + m_1)\dots\Gamma(\alpha_K + m_k)} \prod_{k=1}^K \mu_k^{\alpha_k + m_k - 1}
$$

## Gaussian distribution

Univariate Gaussian distribution:

$$
\mathcal{N}(x \mid \mu, \sigma^2) = \frac{1}{(2\pi\sigma^2)^{1/2}}
\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

Where $\mu$ and $\sigma^2$ are the mean and variance of the population.

Multivariate Gaussian distribution:

$$
\mathcal{N}(\bar x \mid \bar\mu, \Sigma) = 
\frac{1}{(2\pi)^{D/2} |\Sigma|^{1/2}}
\exp\left(
    -\frac{ (\bar x - \bar\mu)^T \Sigma^{-1} (\bar x - \bar\mu) }{2}\right)
$$

Where $D$ is the dimensionality of $\bar x$, $\mu$ is the mean vector and $\Sigma$ is the covariance matrix.

**Central Limit Theorem**. Subject to certain mild conditions, the sum of a set of random variables, which is of course itself a random variable, has a distribution that becomes increasingly Gaussian as the number of terms in the sum increases.

**Observation n.1** - The covariance matrix $\Sigma$ is always positive semi-definite. This means that the eigenvalues are non-negative.

**Observation n.2** - To be well-defined, a Gaussian must have a positive *definite* covariance matrix, which means that all the eigenvalues are strictly positive. If the covariance matrix has one or more null eigenvalues (positive *semi*-definite), then the distribution is singular and is confined to a *subspace of lower dimensionality*. 

**Observation n.3** - Given a 2D Gaussian distribution, we find that elements $x$ with constant density are distributed as ellipses, where the axis of the ellipse is given by the eigenvectors, and the length of the axis is proportional to the square root of the corresponding eigenvalue. The ellipse defined by the axis having a length equal to the square root of the eigenvalues, we find all the elements with a density of $\exp(-1/2)$. 

> This is a hint on my preferred interpretation of eigenvectors and eigenvalues calculated from data: eigenvectors represent that capture most of the variability, and the corresponding eigenvalues are an indicator of the variability on that axis (see PCA).

![2D gaussian](assets_ch2/2dgaussian.png)

**Observation n.4** - A multivariate Gaussian can be decomposed as a product of multiple univariate Gaussians.

**Observation n.5** - The book provides a formal proof to find that $\mathbb{E}[\bar x] = \bar \mu$ and $\text{var}[\bar x] = \Sigma$. 

**Observation n.6** - A general symmetric covariance matrix $\Sigma$ has $D(D+1)/2$ independent parameters, and there are another $D$ independent parameters in $\mu$, giving $D(D+3)/2$ parameters in total. This means that the number of parameters grows quadratically with the dimension $D$.

> One way to reduce the number of parameters is to use restricted forms of the covariance matrix. For example, with a diagonal covariance matrix (figure b) we a linear dependency between parameters and dimensionality, but we explain the variability only from the features axis. Another possibility is to use the **isotropic covariance** (Figure c) $\sigma^2 I$ where we have only one parameter, but we discard the variability along multiple axis. We have a trade-off between model complexity and flexibility that must be adressed based on the application.

![Alt text](assets_ch2/gaussian_parameters.png)

**Limitations of the Gaussian** - Gaussian is unimodal (only 1 maximum) and thus is not good at representing multimodal data.


### Conditional & Marginal Gaussian Distributions

Given a joint Gaussian distribution $\mathcal{N}(x \mid \mu, \Sigma$ with $\Lambda = \Sigma^{-1}$ (precision matrix) and

$$
x = \begin{bmatrix} x_a \\ x_b \end{bmatrix} \hspace{1cm}
\mu = \begin{bmatrix} \mu_a \\ \mu_b \end{bmatrix} 
$$

$$
\Sigma = \begin{bmatrix}
\Sigma_{aa} & \Sigma_{ab} \\
\Sigma_{ba} & \Sigma_{bb} \\
\end{bmatrix}
\hspace{1cm}
\Lambda = \begin{bmatrix}
\Lambda_{aa} & \Lambda_{ab} \\
\Lambda_{ba} & \Lambda_{bb} \\
\end{bmatrix}
$$

where $\Sigma_{ab} = \Sigma_{ba}^T$ and $\Lambda_{ab} = \Lambda_{ba}^T$. 

Then we have that the conditional distribution $p(x_a \mid x_b)$ is

$$
\begin{split}
p(x_a \mid x_b) &= \mathcal N(\mu_{a \mid b}, \Lambda_{aa}^{-1}) \\
\mu_{a \mid b} &= \mu_a + \Sigma_{ab}\Sigma_{bb}^{-1}(x_b - \mu_b)
\end{split}
$$

> Note that $\mu_{a\mid b}$ is a linear function of $x_b$.

where 

$$
\Lambda_{aa}^{-1} = \Sigma_{aa} - \Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba}
$$

And that the marginal distribution $p(x_a)$:

$$
p(x_a) = \mathcal N (x_a \mid \mu_a, \Sigma_{aa})
$$

All the derivations focus on the quadratic relationship of the exponential factor to the $x$ and are detailed starting from page 85. The point is that the conditional and marginal distribution of a joint Gaussian distribution are again Gaussian distributions.

Given a marginal Gaussian distribution for $x$ and $y$ and a conditional Gaussian distribution for $y$ given $x$ in the form:

$$
\begin{split}
p(x) = \mathcal{N}(x \mid \mu, \Lambda^{-1}) \\
p(y \mid x) = \mathcal{N}(Ax+b, L^{-1})
\end{split}
$$

Where $Ax+b$ expresses the fact that the mean of the conditional distribution $y$ given $x$ is a linear function of $x$, and $L^{-1}$ is another precision matrix. The marginal distribution of $y$ and the conditional distribution of $x$ given $y$ are given by:

$$
\begin{split}
p(y) &= \mathcal N (y \mid A\mu + b, L^{-1} + A \Lambda A^T) \\
p(x \mid y) &= \mathcal N (x \mid \Sigma[A^TL(y-b)+\Lambda\mu], \Sigma) 
\end{split}
$$

where 

$$
\Sigma = (\Lambda + A^TLA)^{-1}
$$

### Maximum likelihood for the Gaussian

Let $X = (x_1, \dots, x_N)^T$ be a dataset of observation $x_n$ drawn independently from a multivariate Gaussian distribution. We can estimate the parameters by maximizing the log-likelihood:

$$
p(X \mid \mu, \Sigma) = -\frac{ND}2 \ln(2\pi) - \frac N2 \ln|\Sigma| - \frac12 \sum_{n=1}^N (x_n - \mu)^T \Sigma^{-1}(x_n - \mu)
$$

By setting the derivative to zero, we can compute:

$$
\mu_{ML} = \frac1N\sum_{n=1}^N x_n 
\hspace{1cm}
\Sigma_{ML} = \frac1N\sum_{n=1}^N(x_n-\mu_{ML})(x_n-\mu_{ML})^T
$$

Where we calculate the two parameters in sequential steps since there's no dependency on $\Sigma$ when maximizing with respect to $\mu$. By taking the expectation we see that:

$$
\mathbb E[\mu_{ML}] = \mu \hspace{1cm}
\mathbb E[\Sigma_{ML}] = \frac{N-1}{N}\Sigma
$$

We see that $\Sigma_{ML}$ is a biased estimator, and we can correct it by:

$$
\tilde{\Sigma}_{ML} = \frac1{N-1}\sum_{n=1}^N(x_n-\mu_{ML})(x_n-\mu_{ML})^T
$$

Now $\mathbb E[\tilde\Sigma_{ML}] =\Sigma$ is a correct extimator for the true covariance.

## Student's t distribution

In order to estimate the precision $\tau$ of a Gaussian distribution $\mathcal N(x \mid \mu, \tau^{-1})$ by using a Bayesian approach, we can use the Gamma distribution as a prior:
$$
Gam(\tau \mid a, b) = \frac{b^a}{\Gamma(a)} \tau^{a-1} \exp(-b\tau)
$$
Then we can marginalize to obtain 
$$
p(x \mid \mu, a, b) = \int_0^{\infty} \mathcal N(x \mid \mu, \tau^{-1}) Gam(\tau \mid a, b) d\tau
$$
By performing a number of steps, we can say that this is a Student's t distribution:
$$
St(x \mid \mu, \lambda, \nu) = \frac{\Gamma(\frac\nu2 +\frac12)}{\frac\nu2} \sqrt{\frac{\lambda}{\pi\nu}} \left[1+\frac{\lambda(x-\mu)^2}{\nu}\right]^{-\frac\nu2 - \frac12}
$$
Where $\lambda = a /b$ (precison) and $\nu = 2a$ (degrees of freedom). The precision of the t distribution doesn't correspond to the inverse of the variance! 

The parameters can be estimated by Expectation-Maximization, and the result has the property of **robustness**: outliers does not severely affect the distribution. 

The multivariate version of the t distribution is:
$$
St(x \mid \mu, \Lambda, \nu) = 
\frac{\Gamma(\frac D2 + \frac\nu2)}{\frac\nu2}
\frac{\sqrt{|\Lambda|}}{(\pi\nu)^{D/2}}
\left[ 1 + \frac{\Delta^2}{\nu} \right]^{-\frac D2 - \frac \nu2}
$$
and 
$$
\Delta^2 = (x-\mu)^T\Lambda(x-\mu)
$$
The statistics are

$$
\mathbb{E}[x] = \mu \text{ if } \nu > 1 \\
cov[x] = \frac{\nu}{\nu-2}\Lambda^{-1} \text{ if } \nu > 2\\
mode[x] = \mu
$$

### Exponential family

A distribution that is part of the exponential family can be represented as:

$$
p(x \mid \eta) = h(x) g(\eta) \exp(\eta^T u(x))
$$

Where $\eta$ are the **natural parameters** of the distribution. The function $g(\eta)$ ensures that the distribution is normalized:

$$
g(\eta) \int h(x) \exp(\eta^T u(x)) dx = 1
$$

The Bernoulli ($\eta=\mu$), Multinomial ($\eta = \langle \mu_1, \dots, \mu_K \rangle$) and Gaussian distributions $(\eta = \langle \mu, \sigma \rangle)$ are part of this family, and the PRML book proofs this at page 113.

If we want to estimate $\eta$, we can do that by maximum-likelihood. Let's set the gradient of $p(x \mid \eta)$ w.r.t. $\eta$ to 0:

$$
\nabla \left[ (g(\eta) \int h(x) \exp(\eta^T u(x)) dx \right] = 0 \\

\nabla g(\eta) \underbrace{\int h(x) \exp(\eta^T u(x)) dx}_{\text{(b)}} + g(\eta) \underbrace{\int h(x) \exp(\eta^T u(x)) u(x) dx}_{\text{(a)}} = 0
$$

>(a) To understand the underlined expression, remember that the integration is w.r.t. $x$, while the differentiation is w.r.t. $\nabla$, so we directly differentiate the content as the derivative of sums is the sum of derivatives. 

> (b) recall that $g(\eta) \int h(x) \exp(\eta^T u(x)) dx = 1$, this means that 
> $\int h(x) \exp(\eta^T u(x)) dx = 1/g(\eta)$


$$

-\frac1{g(\eta)} \nabla g(\eta) =  \int h(x)g(\eta) \exp(\eta^T u(x)) u(x) dx \\
-\frac1{g(\eta)} \nabla g(\eta) =  \int p(x \mid \eta) u(x) dx \\
\underbrace{-\frac1{g(\eta)} \nabla g(\eta)}_{\text{(c)}} =  \mathbb{E}[u(x)] \\
- \nabla \ln g(\eta) =  \mathbb{E}[u(x)]
$$

> Point (c) is provided by the identity $\nabla \ln g(\eta) = \frac{1}{g(\eta)}\nabla g(\eta)$ (chain rule). 

We will use this result later on. Now suppose to have $N$ i.i.d. observations drawn from the exponential distribution $X = \{ x_1, \dots, x_N \}$. The likelihood function is given by:

$$
p(X \mid \eta ) = (\prod_{n=1}^N h(x_n)) g(\eta)^N \exp\bigg\{ \eta^T \sum_{n=1}^N u(x_n) \bigg\}
$$

Setting $\nabla \ln p(X \mid \eta) = 0$ we get:

$$
-\nabla \ln g(\eta_{ML}) = \frac 1N \sum_{n=1}^N U(x_n)
$$

which can be solved to obtain $\eta_{ML}$. The solution depends on the data only through $\sum_n u(x_n)$, which is called the **sufficient statistic** of the exponential distribution. For $N \to \infty$ then $\eta_{ML} \to \eta$, 

For each exponential distribution of the previous form, there exists a conjugate prior distribution over the parameters $\eta$ of the following form 

$$
p(\eta \mid \chi, \nu) = f(\chi, \nu) g(\eta)^{\nu} \exp\left\{ \nu\eta^T \chi \right\}
$$

Where $f(\chi, \eta)$ is a normalization coefficient, $g(\eta)$ is the same function presented in the exponential distribution, $\nu$ can be interpreted as a effective number of pseudo-observations in the prior, each of which has a value for the sufficient statistic $u(x)$ given by $\chi$. 

> Again - why do we need conjugate priors??
>
> A prior $p(\eta)$ which is conjugate to the likelihood $p(X \mid \eta)$ produce a posterior $p(\eta \mid X)$ that has the same functional form as the choosen prior. This allows to derive a closed-form expression for the posterior distribution (otherwise you need to compute the normalization coefficient by integration, YOU DON'T WANT TO DO THAT, RIGHT?)

### Noninformative prior

If we have no prior information, we want a prior with minimal influence on the inference. We call such a prior a noninformative prior. The Bayes/Laplace postulate, stated about 200 years ago says the following:

> **The principle of insufficient reason**. When nothing is known about $\theta$ in advance, let the prior $\pi(\theta)$ be a uniform distribution, that is, let all possible outcomes of $\theta$ have the same probability.

One noninformative prior could be the uniform distribution, but there are two problems:

1. If the parameter $\theta$ is unbounded, the prior distribution cannot be correctly normalized because the integral over $\lambda$ diverges. In that case, we have an **improper** prior. In pratice, improper priors are used if the posterior is proper, i.e. is correctly normalized. 
2. The second problem is that if we perform a non-linear change of variable, then the resulting density will not be constant (recall the Jacobian multiplier).


 ## Nonparametric Methods

 The distribution we have seen are governed by parameters that are estimated from the data. This is called **parametric approach** to density modelling.

In this section we talk about nonparametric approaches to density estimation (only simple frequentist methods).

Consider a continuous variable $x$, the simplest way to model that distribution is to partition $N$ observations of $x$ in different bins of width $\Delta_i$ (often the same for every bin $\Delta_i = \Delta$), and then count the number $n_i$ of observation of $x$ falling in bin $i$. To turn this count into a normalized probability density:

$$
p_i = \frac{n_i}{N \Delta_i}
$$

Problems:

* If we choose $\Delta$ too small, the resulting distribution will be too spiky (i.e. will show structures that are not present in the real distribution)
* If we choose $\Delta$ too big, the resulting distribution may fail to capture the structures of the real distribution

![Alt text](assets_ch2/nonparametric.png)

Advantages:

* Good visualization of the distribution
* The dataset can be discarded once the histogram is built
* Good setup if data points are arriving sequentially

Disvantages:

* The estimated density has discontinuities in the bin edges
* Does not scale with dimensionality (curse of dimensionality), the amount of data needed to work in high dimensional spaces is prohibitive.

Good ideas:

* To estimate probability density at a particular location, we should consider the data points that lie within some local neighbourhood of that point.



