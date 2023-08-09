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






