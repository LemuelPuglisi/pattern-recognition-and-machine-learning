# Introduction

This chapter cover the basics concepts. Only hot takes and notes on slightly more complex concepts will be reported.

## Polynomial Curve Fitting Example

Hot takes from the "Polynomial Curve Fitting" example:

+ The squared error loss function is squared with respect to the polynomial weights. This means that its derivative is linear with respect to the polynomial weights and can be solved in a closed form. By finding a unique solution that minimizes the loss, we can determine the optimal polynomial curve fit.
+ A higher degree \\(M\\) corresponds to increased flexibility in the polynomial, but it also makes the model more susceptible to overfitting. Overfitting occurs when the model captures noise or random fluctuations in the training data, leading to poor generalization to new data.
+ To mitigate overfitting, regularization can be employed. Regularization limits the magnitude of the parameters by incorporating a penalty term in the loss function. When the penalty term is quadratic, the regression is known as ridge regression.
+ The \\(\lambda\\) term in the regularization controls the complexity of the model. A higher \\(\lambda\\) value results in a more constrained model with reduced flexibility, helping to prevent overfitting.
+ Another approach to reducing overfitting is by increasing the size of the training set. By including more diverse examples in the training data, the model can learn more general patterns and avoid over-relying on specific instances.

## Probability theory

In page 18, on the paragraph about probability densities, author states that under a non-linear change of variable \\(g\\), a probability density transforms differently from a simple function, due to the Jacobian factor. To understand it better, let's report the answer provided in [math.stackexchange.com](https://math.stackexchange.com/questions/3749123/what-is-the-jacobian-factor).

> Conceptually the idea is the following: if \\(f\\) is a probability density function, it satisfies certain properties, like \\(f \ge 0\\) and
> 
> \\[ \int_{-\infty}^{\infty} f(x)dx = 1 \\]
>
> If we look at a transformation \\(f(g(y))\\), firstly the second property might be not true anymore. Hence, \\(f(g(\cdot))\\) is likely no probability density function. Secondly, what we roughly try to describe with \\(f(g(\cdot))\\) is the probability distribution of a random variable \\(Y\\) that is given such that \\(g(Y)\\) follows the distribution represented by \\(f\\). Say \\(g\\) is invertible and sufficiently smooth. The distribution of \\(Y\\) is given by
>
> \\[ P(Y \in A) = P(g(Y) \in g(A)) = \int_{g(A)} f(x)dx \\]
>
> Which is not very useful in practice, as this integral is on sets of the form \\(g(A)\\). According to the integration by substitution formula, we can compute a probability density function ℎ such that
>
> \\[ P(Y \in A) = \int_A h(y)dy \\]
>
> Here, 
>
> \\[ h = f(g(\cdot)) \det J_g(\cdot) \\]
>
> where \\(J_g\\) is the Jacobian of \\(g\\). 

A consequence of this observation is that the maximum of a probability density is dependent on the choice of variable.



## Exercises 


### Solution to exercise 1.1



Given 

\\[
E(\bar w) = \frac12 \sum_{n=1}^N \left( 
x_n^M w_M + \dots + x_n w_1 + w_0 - t_n 
\right)^2
\\]

Compute the partial derivative

\\[
\frac{\partial E}{\partial w_i} = 
\sum_{n=1}^N x_n^i \left(
x_n^M w_M + \dots + x_n w_1 + w_0 - t_n 
\right)
\\]

Set the partial derivatives to zero to get $w$ that minimizes \\(E\\)

\\[
\frac{\partial E}{\partial w_i} = 
\sum_{n=1}^N x_n^i \left(
x_n^M w_M + \dots + x_n w_1 + w_0 - t_n 
\right) = 0
\\]

\\[
= \sum_{n=1}^N
x_n^{M+i} w_M + \dots + x_n^{1+i} w_1 + x_n^{0 + i}w_0 - x_n^{i} t_n = 0 
\\]

\\[
= \sum_{n=1}^N
x_n^{M+i} w_M + \dots + x_n^{1+i} w_1 + x_n^{0 + i}w_0 = \sum_{n=1}^N x_n^{i} t_n
\\]

\\[
= w_M \sum_{n=1}^N x_n^{M+i} + \dots + w_j \sum_{n=1}^N x_n^{j+i} + \dots +
 w_0 \sum_{n=1}^N x_n^{0 + i} = \sum_{n=1}^N x_n^{i} t_n
\\]

\\[
= A_{iM} w_M + \dots + A_{ij} w_j + \dots + A_{i0} w_0  = T_i 
\\]

\\[ = \sum_{j=1}^M A_{ij}w_j = T_i \\]