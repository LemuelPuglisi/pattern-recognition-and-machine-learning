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

 ## Decision theory

Probability theory is useful to quantify uncertainty, while decision theory allows to make optimal decisions in situations involving uncertainty.



### Medical diagnosis example

Consider a medical diagnosis problem in which we have taken an X-ray image of a patient. Our algorithm has to predict if the patient has cancer or not. The input vector $\bar x$ is the set of pixel intensities, while the output variable $t$ is binary ($C_1=0$ no cancer, $C_2=1$ cancer). 

The **general inference problem** involves determining the joint distribution $p(\bar x, t)$. Given it, we then must **decide** to give treatments to the patient or not, and we would like this choice to be optimal. This is called the **decision step**.

If our goal is to make as few misclassifications as possible, then it is sufficient to study the posterior probability $p(C_k \mid \bar x)$. Assigning each observation to a class means dividing the input space in different **decision regions** $R_k$ ($k=1, 2$) where the boundaries are called **decision boundaries / surfaces**. Regions are not constrained to be contiguous but can be comprised of different disjoint sub-regions. 

Given our decision regions, the probability of making a mistake is quantified by:
\\[
\begin{split}
p(\text{mistake}) &= p(\bar x \in R_1, C_2) + p(\bar x \in R_2, C_1) \\\\
&= \int_{R_1} p(\bar x, C_2) d\bar x +  \int_{R_2} p(\bar x, C_1) d \bar x \\\\
&= \int_{R_1} p(C_2 \mid \bar x)p(\bar x) d\bar x +  \int_{R_2} p(C_1\mid \bar x)p(\bar x) d \bar x
\end{split}
\\]
Since the prior $p(\bar x)$ is common in both terms, we can say that the minimum probability of making a mistake is given if each observation $\bar x$ is assigned to the class $C_k$ for which the posterior probability $p(C_k \mid \bar x)$ is the largest.



> For the case of $K$ classes, is easier to maximize the probability of being correct:
> $$
> p(\text{correct}) = \sum_{k=1}^K p(\bar x \in R_k, C_k) = \sum_{k=1}^K \int_{R_k}p(\bar x, C_k)d\bar x = \sum_{k=1}^K \int_{R_k}p(C_k \mid \bar x)p(\bar x)d\bar x
> $$
> The maximum probability is obtained when each observation is assigned to the class with highest posterior.



### Minimizing the expected loss

In our example, we have two different misclassification:

+ A healthy patient being classified as having cancer (which is bad)
+ A patient with cancer being classified as healthy (which is worse due to late treatments)

We can formalize the severity of each misclassification by constructing a **loss function** (or cost function) which is a single measure of loss incurred in taking any of the available decisions or actions. Let's construct a **loss matrix** $L$:
$$
L = \begin{bmatrix}
0 & 1\\
1000 & 0 \\
\end{bmatrix}
$$
Where $L_{ij}$ indicates the loss severity of classifying an observation of class $i$ with class $j$. The diagonal indicates correct classifications, $L_{21}=1000$ is the loss of classifying a patient with cancer ($C_2$) to be healthy ($C_1$), vice versa for $L_{12}=1$.   

The optimal solution is the one which minimizes the average loss function:
$$
\mathbb{E}[L] = \sum_{k}\sum_{j}\int_{R_j} L_{kj} p(\bar x, C_k) d\bar x
$$
Our goal is to choose the regions $R_j$ to minimize the expected loss, which can be formulated with priors instead of the joint probability. Again, the decision rule which minimize the loss is the one that assigns each observation to the class with highest posterior.

If the posterior is too low or comparable with the other posteriors, our classification has higher uncertainty of being true. We can use a threshold $\theta$ such that if $p(C_k \mid \bar x) \le \theta$ then we avoid making a decision. This is known as the **reject option**.



### Inference and decision

The classification is now broken in two stages:

+ Inference stage, where we learn the model $p(C_k \mid \bar x)$
+ Decision stage, where we use posteriors to make optimal assignments 

We can identify also 3 approaches to solve the decision stage:

1. **Generative models**. Model the joint distribution $p(\bar x, C_k)$, obtain posteriors, make decisions based on posteriors. The joint distribution is useful to simulate samples from the modeled population and for outliers / novelty detection.
2. **Discriminative models**. Determine only the posterior class probabilities and assign each observation to the class with highest posterior accordingly. 
3. **Discriminative functions**. Model a function $f(\bar x) = C_k$ directly, where probabilities play no role. 

The benefits of computing the posterior probabilities (avoiding approach n.3) are:

+ We can modify the loss matrix without re-training the model (minimizing risk)
+ We can use the reject option
+ When balancing the dataset, we can compensate for class priors (**A**)
+ We can combine models (**B**)

>(**A**) Suppose we have 1000 samples from class $C_1$ and 100 samples from class $C_2$, so the real priors are $p(C_1) = 0.91$ and $p(C_2) =  0.09$. Suppose we want to balance our datasets to 100 samples for each class. We know the real priors and hence we can replace them when using the Bayes theorem.
>
>(**B**) Suppose we have X-ray data $\bar x_I$ and blood test data $\bar x_B$, we can develop two models (one for each) and assume that the distributions of the inputs are independent given the class (conditional independence)
>$$
>p(\bar x_I, \bar x_B \mid C_k) = p(\bar x_I \mid C_k) p(\bar x_B \mid C_k)
>$$
>The posterior is then
>$$
>\begin{split}
>P(C_k \mid \bar x_I \bar x_B) &\propto p(\bar x_I, \bar x_B \mid C_k) p(C_k)\\
>&\propto p(\bar x_I \mid C_k) p(\bar x_B \mid C_k)p(C_k) \\
>&\propto \frac{p(C_k \mid \bar x_I)p(C_k \mid \bar x_B)}{p(C_k)}
>\end{split}
>$$
>Hence the combination of 2+ models is trivial.



### Loss functions for regression

The decision stage for regression problems consists of choosing a specific estimate $ y(x) $ of the value $t$ for each input $x$. In doing so, we incur a loss $L(t, y(x))$. The expected loss is then given by:

$$
\mathbb{E}[L]= \int\int L(t, y(x)) p(x, t) dx dt
$$

A common choice of loss is the squared loss

$$
\mathbb{E}[L]= \int\int \{y(x) - t\}^2 p(x, t) dx dt
$$

Supposing that $y$ is a flexible function, this can be solved by using the calculus of variations, obtaining the following regression function

$$
y(x) = \int t \cdot p(t \mid x) dt = \mathbb{E}_t [t \mid x]
$$

## Notes on Information theory section

Let $p$ and $q$ be the exact and an approximate distributions of the random variable $x$. The KL divergence defines the additional amount of information (in nats for $\ln$ bits for $\log_2$) required to transmit $x$ assuming its distribution is $q$ (approximated) instead of $p$ (exact). The mathematical definition is:

$$
KL(p || q) = H(p || q) - H(p) = -\int p(x) \ln \frac{q(x)}{p(x)} dx
$$

From the Jensen Inequality, we know that for a convex function $f$

$$
f(\mathbb{E}[x]) \le \mathbb{E}(f(x))
$$

Since $-\ln$ is a strictly convex function, we can apply the Jensen Inequality to the KL divergence:

$$
\begin{split}
KL(p || q) &= -\int p(x) \ln \frac{q(x)}{p(x)} dx \\
& \ge -\ln \int p(x) \frac{q(x)}{p(x)} dx \\
& \ge -\ln \int q(x) dx  = - \ln 1 = 0\\
\end{split}
$$

proceed from here...