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


## Bayesian curve fitting

Let's revise the polynomial curve fitting example to also estimate uncertainty. We suppose that each prediction is normally distributed with the mean centered at the prediction, and the variance \\(\beta^{-1}\\) estimated along with the polynomial weights \\(\bar w\\):
\\[
p(t \mid x, \bar w, \beta) = \mathcal N (t \mid y(x, \bar w), \beta^{-1})
\\]
Supposing that the \\(N\\) points from the dataset \\(\langle X, T \rangle\\) are drawn independently, the likelihood function of a prediction is given by:
\\[
p(T \mid X, \bar w, \beta) = \prod_{n=1}^N \mathcal N (t_n \mid y(x_n, \bar w), \beta^{-1})
\\]
It is simpler and numerically convenient to work with the log-likelihood:
\\[
\ln p(T \mid X, \bar w, \beta) =
-\frac{\beta}2 \sum_{n=1}^N (y(x_n, \bar w) - t_n)^2 +
\frac{N}2\ln \beta - \frac{N}2\ln(2\pi)
\\]
We should find \\((\bar w, \beta)\\) (\\(\beta\\) is the precision, inverse of variance) which maximize the likelihood. If we consider first the parameters \\(\bar w\\), then the maximization problem can drop the last 2 terms since they do not depend on \\(\bar w\\), which is the same as minimizing the sum of squares error loss.

Once found \\(\bar w_{ML}\\), we can find \\(\beta\\) as
\\[
\frac{1}{\beta_{ML}} = \frac1N \sum_{n=1}^N (y(x_n, \bar w) - t_n)^2
\\]
By using \\(\bar w\\) and \\(\beta\\), instead of providing a single prediction, we can provide a full distribution \\(p(t \mid x, \bar w_{ML}, \beta_{ML})\\) over the values of \\(t\\) for each \\(x\\). 



Let \\(p(\bar w \mid \alpha)\\) be the prior distribution over the weights \\(\bar w\\), for simplicity:
\\[
p(\bar w \mid \alpha) = \mathcal N(\bar w \mid \bar 0, \alpha^{-1} I)
\\]
Where the hyperparameter \\(\alpha\\) is the precision of the distribution. Using the Bayes theorem, the posterior distribution \\(p(\bar w \mid X, T, \alpha, \beta)\\) is proportional to the product of the likelihood and the prior:
\\[
p(\bar w \mid X, T, \alpha, \beta) \propto p(T \mid X, \bar w, \beta) p(\bar w \mid \alpha)
\\]
By now we can find the most probable weights \\(\bar w\\) by maximizing the posterior distribution. This approach is called MAP (MAximum Posterior). We find that maximizing the posterior defined previously is the same as minimizing
\\[
\frac{\beta}{2}\sum_{n=1}^N (y(x_n, \bar w) - t_n)^2 + \frac\alpha2\bar w^T \bar w
\\]
 So maximizing the posterior is the same as minimizing the **regularized** sum of squares error loss function, where the regularization parameter is \\(\lambda = \frac\alpha\beta\\).

For a fully Bayesian treatment, we should evaluate \\(p(t \mid x, X, T)\\). This requires to integrate over all the possible \\(\bar w\\). By using the product and sum rules, we can write:
\\[
p(t \mid x, X, T) = \int p(t \mid x, \bar w, \beta) 
p(\bar w \mid X, T, \alpha, \beta)  d\bar w
\\]
Which can be solved analytically, finding that:
\\[
p(t \mid x, X, T) = \mathcal N (t \mid m(x), s^2(x))
\\]
where 
\\[
m(x) = \beta \phi(x)^T S \sum_{n=1}^N \phi(x_n) t_n
\\]
and 
\\[
s^2(x) = \beta^{-1} + \phi(x)^T S \phi(x)
\\]
and 
\\[
S^{-1} = \alpha I + \beta \sum_{n=1}^N \phi(x_n) \phi(x_n)^T
\\]
where \\(\phi(x) = \langle 1, x, x^2, \dots, x^M\rangle\\). 