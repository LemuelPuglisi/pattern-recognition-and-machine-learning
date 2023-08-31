# Linear models for regression

The goal of regression is to predict the value of one or more continuous targets variables $t$ given the value of a D-dimensional vector $x$ of input variables. 

By linear models we mean that the model is a linear function of the adjustable parameters. E.g. the polynomial curve fitting algorith builds a linear model. The simplest form of linear regression models are also linear functions of the input variables. 

We get a more useful class of functions by taking linear combinations of a fixed set of nonlinear functions of the input variables, known as **basis functions**. Such models are linear functions of the parameters (which gives simple analytical properties) and yet can be nonlinear with respect to the input variables.

Given a dataset of $N$ observations $\{x_n\}$ where $n=1, \dots, N$, together with the corresponding target values $\{t_n\}$, the goal is to predict $t$ for a new value of $x$. 

* **Simple approach**: Find an appropiate function $y(x) \approx t$
* **General approach**: Find the predictive distribution $p(t \mid x)$ to get the uncertainty of a prediction

## Linear Basis Function Models

The simplest linear model involves a linear combination of the input variables, also called linear regression:

$$
y(x, w) = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_D x_D
$$

This is:

* A linear function of the parameters (good for optimization)
* A linear function of the inputs (bad for expressiveness)

Extend the concept of linear combination to combine fixed nonlinear functions of the input:

$$
y(x, w) = w_0 + \sum_{j=1}^{M-1} w_j \phi_j(x)
$$

where $\phi_j(x)$ are known as **basis functions**. By having $M-1$ components, the total number of parameters is $M$ (consider the bias). 

If we consider $\phi_0(x) = 1$, then we can write:

$$
y(x, w) = \sum_{j=0}^{M-1} w_j \phi_j(x)
$$

These linear models are:

* A linear function of the parameters (good for optimization)
* A nonlinear function of the inputs (good for expressiveness)

Polynomials are basis functions of the form $\phi_j(x) = x^j$. The problem with polynomial is that they are global functions: a change in a region of the input space affects all the other regions. 

Other choices for the basis functions are:

$$
\phi_j(x) = \exp \bigg\{ - \frac{(x-\mu_j)^2}{2s^2} \bigg\}
$$

Which is the Gaussian basis function. In this case, $\mu_j$ is the location in the input space, and $s$ is the scale. This function doesn't have a probabilistic interpretation.

Another possibility is the Sigmoidal basis function:

$$
\phi_j(x) = \sigma\left(\frac{x-\mu_j}{s}\right)
\hspace{1cm}
\sigma(a) = \frac{1}{1+\exp(-a)}
$$

Where $\sigma$ is the logistic function, but we can also use the tanh function.

> We can use also Fourier basis functions such that the the regression function is an expansion of sinusoidal functions at a certain frequency. Combining basis functions localized in both space and frequency leads to a class of functions known as wavelets.

## Maximum likelihood and least squares

Let $y$ be a deterministic function such that $y(x) \approx t$, let $\epsilon \sim \mathcal N(\epsilon \mid 0, \beta^{-1})$ be a random Gaussian variable with precision $\beta$. We assume that the target variable $t$ is given by:

$$
t = y(x, w) + \epsilon
$$

The conditional distribution of $t$ will then be 

$$
p(t \mid x) = \mathcal{N}(t \mid y(x,w), \beta^{-1})
$$

For a new value $x$, the optimal prediction of $t$ is given by the conditional mean:

$$
\mathbb{E}[t \mid x] = \int t p(t \mid x)dt = y(x, w)
$$

For a dataset $X = \{ (x_n, t_n)\}_{n=1}^{N}$, let $T = [t_1, \dots, t_N]^T$, assuming that $y(x, w)$ is given by a linear model $y(x, w) = w^t \phi(x)$, then the likelihood of the target variables is given by:

$$
p(T \mid X, w, \beta) = \prod_{n=1}^{N} \mathcal{N}(t_n \mid w^T\phi(x_n), \beta^{-1}
$$

The log-likelihood is:

$$
\ln p(T \mid X, w, \beta) = \frac{N}{2}\ln \beta - \frac{N}{2}\ln (2\pi) - \beta E_D(w) 
$$

Where $E_D$ is the sum-of-squares error function:

$$
E_D(w) = \frac12\sum_{n=1}^N (t_n - w^T\phi(x_n))^2
$$

We now estimate the parameters $\beta, w$ by maximum likelihood. The gradient w.r.t. to $w$ is:

$$
\nabla \ln p(T \mid X, w, \beta) = \sum_{n=1}^N (t_n - w^T\phi(x_n))\phi(x_n)^T
$$

By setting the gradient to 0 and solving for $w$ we find:

$$
w_{ML} = (\Phi^T \Phi)^{-1} \Phi^T T
$$

which are known as the **normal equations** for the least squares problem. Here $\Phi$ is a NxM matrix called **design matrix** whose elements are given by $\Phi_{nj} = \phi_j(x_n)$.

The quantity 

$$
\Phi^\dagger = (\Phi^T \Phi)^{-1} \Phi^T
$$

Is known as Moore-Penrose pseudo-inverse of the matrix $\Phi$, which is a generalization of the inverse for nonsquare matrices.

If we solve for the bias parameter $w_0$, the solution suggests that the bias compensates for the difference between the averages (over the training set) of the target values and the weighted sum of the averages of the basis function values.

Maximizing for the precision parameter we get:

$$
\frac{1}{\beta_{ML}} = \frac1N \sum_{n=1}^N \{ t_n - w_{ML}^T \phi(x_n) \}^2
$$

which is basically the precision of the residuals.

## Geometric interpretation of least square solution

Consider an N-dimensional space. Let $\mathbb t $ be a vector in that space, where the N components are the ground truth target variables for all the N observations we are trying to predict. 

Build a vector $\mathbb{t} = [t_1, \dots, t_N]^T$ made of the target variables of our dataset made of N observation. This vector lives in a N-dimensional space.

The input variable $x$ is D-dimensional, while we use the basis functions $\phi(x)$ that are M-dimensional. Consider each component of the basis function evaluated on all the N observations of our dataset, we have $M$ vectors in the N-dimensional space which span a subregion S of dimension $M$. 

The target value is predicted by combining the basis function output using some weights $w$, and therefore the N-dimensional vectore $\mathbb y$ made of the predicted target value for each observation in the dataset is indeed a linear combination of the $M$ vectors, and resides inside the subregion $S$. 

The book demonstrates how the solution $\mathbb y$ from the least square problem corresponds to the orthogonal projection of $\mathbb t$ to the closest M-dimensional subregion S.


## Sequential Least Squares

Authors suggest to use gradient descent to get the least square solution sequentially (one observation at the time). Given the sum-of-squares loss, the update of the weights $w$ is 
$$
w^{(\tau + 1)} = w^{\tau} + \eta \bigg[ (t_n - w^{(\tau)T}\phi(x_n)) \phi(x_n)  \bigg]
$$

## Regularized least squares

Adding a regularization term to the loss function helps avoiding overfitting to the data. The most famous regularization term for least squares is weight decay, where the optimization process is forced to produce small weights unless supported by the data. The general form of weight decay is:
$$
\frac\lambda2 \sum_{j=1}^M |w_j|^q
$$
For $q=2$ we have the classic **quadratic regularizer**. For $q=1$ we have the **Lasso regularizer** which has the property (for $\lambda$ sufficiently large) of driving some of the weights to zero, leading to a sparse model. This is useful to avoid overfitting when we have a small dataset, even if the problem becomes to find the suitable $\lambda$. 

## Multiple outputs

Given a regression problem with a multivariate output, the book demonstrates how the solution decouples between the different target variables (they all share the same pseudo-inverse matrix $\Phi^\dagger$ assuming that the target variables are distributed by an isotropic gaussian). Most of the time, we can work with a single variable and easily generalize to the multivariate case.

## Bias-variance decomposition

Suppose we want to find a function $y$ that approximates the target value $y(x) \approx t$ on the input $x$. We model the relation between the input $x$ and the target value $t$ as 
$$
t = f(x) + \epsilon \hspace{1cm} \epsilon \sim \mathcal N(0, \sigma)
$$
We assume that $t$ has random noise, so it's a random variable distributed by
$$
t \sim \mathcal N(y(x), \sigma)
$$
We want to find $y= f$.  Let $L(t, y(x))$ be a loss function that measures the prediction error, then the average loss is:
$$
\mathbb{E}[L] = \iint L(t, y(x)) \cdot p(x, t) dx dt
$$
If the loss is the MSE, then we have:
$$
\begin{split}
\mathbb{E}[L] &= \iint [y(x) - t]^2 \cdot p(x, t) dx dt \\
&= \underbrace{\int [y(x) - \mathbb{E}[t \mid x]] p(x)dx}_{\text{depends on y}} + 
\underbrace{\int [\mathbb{E}[t \mid x] - t] p(x)dx}_{\text{depends on data}} 
\end{split}
$$

> $\mathbb{E}[t \mid x]$ is the expected value of $t$, which is now considered a random variable since we assume it contains random noise. The conditioning on $x$ reflects the fact that the Gaussian distribution is centered at $f(x)$, which depends on $x$.

* The first term depends on $y$ and can be reduced to zero with an unlimited amount of data.
* The second term depends on the noise $\epsilon$ in the data, so it can't be changed by acting on $y$, so it is the minimum achievable value of expected loss.   

Now let's consider K different datasets drawn indipendently from the same distribution $p(x,t)$. We estimate a different function $y$ for each dataset, since they all contain random noise. We can define $\mathbb{E}_D[y(x; D)]$ as
$$
\mathbb{E}_D [y(x; D)] = \frac1K \sum_D y(x; D)
$$
Now consider the square loss and add and subtract the term $\mathbb{E}_D[y(x; D)]$
$$
\{y(x;D) - \mathbb{E}[t \mid x]\}^2 = \\ 
= \{y(x;D) - \mathbb{E}_D[y(x; D)] + \mathbb{E}_D[y(x; D)] - \mathbb{E}[t \mid x]\}^2 = \\
= {\{y(x;D) - \mathbb{E}_D[y(x; D)]}\}^2 + \{\mathbb{E}_D[y(x; D)] - \mathbb{E}[t \mid x]\}^2
+ 2 {\{y(x;D) - \mathbb{E}_D[y(x; D)]}\}\{\mathbb{E}_D[y(x; D)] - \mathbb{E}[t \mid x]\}
$$
If we take the expectation of this term w.r.t. the dataset $D$, then we have:
$$
\mathbb{E}[\{y(x;D) - \mathbb{E}[t \mid x]\}^2] = 
\underbrace{\big\{ \mathbb{E}_D[y(x;D)] - \mathbb{E}[t \mid x] \big\}^2}_{\text{bias}^2} +
\underbrace{\mathbb{E}_D[ \big\{y(x; D) - \mathbb{E}_D[y(x;D)]\big\}^2 ]}_{\text{variance}}
$$
The expected squared difference between the model predictions and the observed data can be expressed as the sum of two terms, the bias squared and the variance.

* The squared bias term represents to which extent the average prediction over all datasets differs from the desired function $\mathbb E[t \mid x]$
* The variance term measures the extent to which the solutions for individual datasets vary around their average (sensitiveness to the choice of dataset)

If we apply this observation to the expected loss value shown before, we have the following decomposition:
$$
\text{expected loss} = (\text{bias})^2 + \text{variance} + \text{noise}
$$
Where
$$
\begin{split}
(\text{bias})^2 &= \int \big\{ \mathbb{E}_D[y(x;D)] - \mathbb{E}[t \mid x] \big\}^2 p(x)dx \\
\text{variance} &= \int \mathbb{E}_D[ \big\{y(x; D) - \mathbb{E}_D[y(x;D)]\big\}^2 p(x)dx \\
\text{noise} &= \int \big\{ \mathbb{E}[t \mid x] - t \big\}^2p(x, t) dxdt
\end{split}
$$

> Mathematically: recall the decomposition of the loss in two terms, we took the first term and further decomposed it into squared variance + variance. The expectation we took is w.r.t. the datasets, but we need to calculate it against the input $x$. 

In practice, bias-variance decomposition can be estimated numerically by replacing the expectation with averages on the observed data. The method requires to have multiple datasets, but that means that all the datasets can be merged in a single big dataset that will produce less overfitted models. Bias-variance decomposition isn't the best way to validate our models, but it's useful to understand how overfitting works. 

## Bayesian Linear Regression

We introduce a Bayesian treatment for linear regression, which will avoid over-fitting and will lead to automatic methods of determining model complexity using training data alone.

### Parameter distribution

The likelihood function is the exponential of a quadratic function of the parameters $w$ (as defined previously)
$$
p(T \mid w) = \prod_{n=1}^N \mathcal{N}(t_n \mid w^t \phi(x_n), \beta^{-1})
$$
Where $T$ are all the target values in the dataset and $\beta$ is the noise precision. Therefore, the conjugate prior over $w$ is given by a Gaussian distribution of the form:
$$
p(w) = \mathcal{N}(w \mid m_0, S_0)
$$
Where $m_0 ,S_0$ are the mean and covariance. 

The posterior $p(w \mid T)$ is a Gaussian distribution (we are using a conjugate prior) proportional to the likelihood and the prior. We calculate the normalization coefficient using the result from 2.116 (from PRML). 
$$
p(w \mid T) = \mathcal{N}(w \mid m_N, S_N)
$$
 Where
$$
m_N = S_N(S_0^{-1}m_0 + \beta \Phi^T T) \\
S_N^{-1} = S_0^{-1} + \beta \Phi^T\Phi
$$
Since the posterior is a Gaussian, its mode coincides with its mean, thus the maximum posterior weight vector is simply given by $w_{map} = m_N$.

**The Bayesian approach is automatically regularized**. Assume the prior to be a zero-mean isotropic Gaussian governed by a single parameter $\alpha$
$$
p(w \mid \alpha) = \mathcal{N}(w \mid 0, \alpha^{-1}I)
$$
The parameters of the posterior distribution will then be given by:
$$
m_N = \beta S_N \Phi^T T \hspace{1cm}
S_N^{-1} = \alpha I + \beta \Phi^T \Phi
$$
The log of the posterior distribution is given by:
$$
\ln p(w \mid T) = -\frac\beta2 \sum_{n=1}^N \{t_n - w^T \phi(x_n)\}^2 - \frac\alpha2 w^tw + \text{const}
$$
The maximization of the posterior is equivalent to the minimization of the sum of squares with the addition of a quadratic regularization term with $\lambda = \alpha / \beta$. 

### Predictive distribution

Once we have the posterior distribution over the weights $w$, how do we estimate the target value $t$ for a new point $x$? We use the **predictive distribution**.
$$
\underbrace{p(t)}_{\text{predictive}} = \int \underbrace{ p(t \mid w) }_{\text{target}} \underbrace{p(w)}_{\text{posterior}} dw
$$
Where we recall that:
$$
\begin{split}
p(t \mid w) &= p(t \mid x, w, \beta) = \mathcal{N}(t \mid y(x,w), \beta^{-1})\\
p(w) &= p(w \mid T) = \mathcal{N}(w \mid m_N, S_N)
\end{split}
$$
The solution to this integral is explained in (2.115). We have
$$
p(t) = \mathcal{N}(t \mid m_N^T \phi(x), \sigma_N^2(x))
$$
where the variance
$$
\sigma_N^2(x) = \underbrace{\frac1\beta}_{\text{noise}} + \underbrace{\phi(x)^TS_N\phi(x)}_{\text{uncertainty}}
$$
Because the noise process and the distribution of $w$ are independent, the variances are additive. For $N \to \infty$, the second term goes to zero, and the variance of the predictive distr. is only given by noise in the data.

The more data we have, the narrower is the predictive distribution, in fact it can be shown that $\sigma_{N+1}^2(x) \le \sigma_{N}^2(x)$ (Qazaz et al., 1997).

### Equivalent kernel

To perform inference using the predictive distribution, we return the mean value, which can be written in the form:
$$
\begin{split}
y(x,m_N) &= m_N^T\phi(x) \\
&= \beta \phi(x)^T S_n \Phi^T T
= \sum_{n=1}^N \beta \phi(x)^T S_n \phi(x_n) t_n
\end{split}
$$
The mean of the predictive distribution is a linear combination of the target variables $t_n$ from the training set:
$$
y(x, m_N) = \sum_{n=1}^N k(x,x_n) t_n \hspace{1cm} k(x,x_n)= \beta \phi(x)^T S_n \phi(x_n)
$$
Where $k(x,x')$ is called **smoother matrix** or **equivalent kernel**. Regression functions that make inference by taking linear combination of the training target values are called **linear smoothers**. Such kernels have a localization property that increase the response if $x$ and $x'$ are closer.

An alternative approach of linear regression is to directly compute an equivalent kernel instead of working with the basis functions. This leads to the Gaussian processes.

Some properties of the kernels are that (1) the weights sum to one $\sum_{n=1}^N k(x,x_n)=1$ and (2) the function can be expressed as an inner product $k(x,z)=\psi(x)^T\psi(z)$ where $\psi$ is a non linear function. 


## Bayesian Model Comparison

Suppose we want to compare $L$ models $M_1, \dots, M_L$, where a model represents a different probability distribution over the observed data $D$. The uncertainty of the model is expressed by a prior distribution $p(M_i)$ (we can assume to be uniform). Given the dataset $D$, we want to evaluate the posterior distribution:

$$
p(M_i \mid D) \propto p(D \mid M_i)p(M_i)
$$

$P(D \mid M_i)$ is called **model evidence** or **marginal likelihood**, since it can be viewed as a likelihood function over the space of models, in which the parameters have been marginalized out.


> The ratio of model evidences $p(D \mid M_i) / p(D \mid M_j)$ is called **Bayes factor**.

Given the posterior $p(M_i \mid D)$, the predictive distribution is given by the sum and product rule:

$$
\begin{split}
p(t \mid x, D) &= \sum_{i=1}^L p(t, M_i \mid x, D) \\
&= \sum_{i=1}^L p(t \mid x, D, M_i) p(M_i \mid x, D) \\
&= \sum_{i=1}^L p(t \mid x, D, M_i) p(M_i \mid D) \\
\end{split}
$$

This is an example of a **mixture distribution**, obtained by averaging the predictive distributions of individual models weighted by the posterior probabilities of those models.


> An approximation of model averaging is to use the most probable model alone to make predictions. This is called **model selection**.

Now we focus on the model evidence / marginal likelihood. For a model $M_i$ governed by parameters $w$, the evidence is:

$$
p(D \mid M_i) = \int p(D \mid w, M_i) p(w \mid M_i) dw
$$

1. The evidence can be viewed as the probability of generating $D$ from $M_i$ by randomly sampling $w \sim p(w \mid M_i)$.
2. The evidence appears as the normalization coefficient in the posterior $p(w \mid D)$