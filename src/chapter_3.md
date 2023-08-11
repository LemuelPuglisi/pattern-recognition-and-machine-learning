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














