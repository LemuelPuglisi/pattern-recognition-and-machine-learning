# Linear models for classifications

## 4.1.3 Least squares for classification

> ### Matrices legend.
> | Matrix      | Dimension        |
> | ----------- | ---------------- |
> | $\tilde W$  | $(D+1) \times K$ |
> | $\tilde{X}$ | $N \times (D+1)$ |
> | $T$         | $N \times K$     |

Consider a classification task with $K$ classes, let $t$ be a one-hot encoding target vector. Each class $C_k$ is described by its own linear model so that

$$
y_k(x) = w_k^T x + w_{k_0} \hspace{1cm} k=1,\dots,K
$$

By using vector notation, we can combine them together:

$$
y(x) = \tilde{W}^T \tilde{x}
$$

Where $\tilde{W}$ is a $(D+1) \times K$ matrix such that the $k$-th column is $(w_{k_0}, w_k^T)^T$ and $\tilde{x}=(1, x^T)^T$.  

Objective: to determine the parameters of $\tilde{W}$ by minimizing a sum-of-squares loss function. 

Consider a training dataset $\{ x_n, t_n \}_{n=1}^{N}$ and define two sparse matrices 
* $T$ of dimension $N \times K$ such that the $n$-th row is the binary one-hot-encoded vector $t_k$. 
* $\tilde X$ of dimension $N \times (D+1)$ such that the $n$-th row is $\tilde{x}_n^T$

The sum-of-squares loss function can be written as:

$$
E_D(\tilde{W}) = \frac12\text{Tr}\left\{
    (\tilde{X}\tilde{W} - T)^T(\tilde{X}\tilde{W} - T)
\right\}
$$

> Question: why do we use the trace?

Set the derivative of $E_D$ w.r.t. $\tilde W$ to zero and obtain the following solution:

$$
\tilde W = (\tilde X^T \tilde X)^{-1} \tilde X^T T = (\tilde X^\dagger)^T T
$$

If we want to obtain the result without using too much matrix calculus we can do the following:

$$
\begin{split}
\tilde X \tilde W &= T \\
\tilde X^T \tilde X \tilde W &= \tilde X^T T \\
(\tilde X^T \tilde X)^{-1}\tilde X^T \tilde X \tilde W &= (\tilde X^T \tilde X)^{-1} \tilde X^T T \\
\tilde W &= (\tilde X^T \tilde X)^{-1} \tilde X^T T \\
\tilde W &= \tilde X^\dagger T \\
\end{split}
$$

The discriminant function will be:

$$
y(x) = \tilde W^T \tilde x = T^T (\tilde X^\dagger)^T \tilde x  
$$

Problems with the discriminant function obtained through minimization of SSE:

* Sensible to outliers
* Bad performances since it estimates $\mathbb{E}[t \mid x]$ under assumption of Gaussian noise, which is clearly wrong when estimating a binary vector $t_n$

### An interesting property

Every target vector in the training set satisfies some linear constraint:

$$
a^T t_n + b = 0
$$

For some costants $\bar a, b$. The model prediction for any value of $x$ will satisfy the same constraint 

$$
a^T y(x) + b = 0
$$

If we use a one-hot-encoding scheme for $t_n$, then components of $y(x)$ will sum up to 1. However, this is not enough for considering $y(x)$ a probability distribution since its components are no bound to $[0,1]$. 
