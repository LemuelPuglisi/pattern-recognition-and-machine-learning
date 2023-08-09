# Linear models for regression

The goal of regression is to predict the value of one or more continuous targets variables $t$ given the value of a D-dimensional vector $x$ of input variables. 

By linear models we mean that the model is a linear function of the adjustable parameters. E.g. the polynomial curve fitting algorith builds a linear model. The simplest form of linear regression models are also linear functions of the input variables. 

We get a more useful class of functions by taking linear combinations of a fixed set of nonlinear functions of the input variables, known as **basis functions**. Such models are linear functions of the parameters (which gives simple analytical properties) and yet can be nonlinear with respect to the input variables.

Given a dataset of $N$ observations $\{x_n\}$ where $n=1, \dots, N$, together with the corresponding target values $\{t_n\}$, the goal is to predict $t$ for a new value of $x$. 

* **Simple approach**: Find an appropiate function $y(x) \approx t$
* **General approach**: Find the predictive distribution $p(t \mid x)$ to get the uncertainty of a prediction

## Linear Basis Function Models

