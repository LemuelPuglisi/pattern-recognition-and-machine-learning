# Probability distributions

This chapter will focus on the problem of density estimation, which consists in finding / estimating the probability distribution $p(x)$ from $N$ independent and identically distributed datapoints $x_1, x_2, \dots, x_N$ drawn from $p(x)$. There are two main ways of doing that: the first way is to use parametric density estimation, where you choose one known parametric distribution (e.g., Gaussian) and try to get the right parameters that fit the data. This method assumes that the parametric distribution we use it's suitable for the data, which is not always the case. Another way of doing that is using non-parametric density estimation techniques (e.g., histograms, nn, kernels).

## Bernoulli experiment

$$
\begin{split}
\ln p(D \mid \mu) &= \ln \prod_{n=1}^N p(x_i \mid \mu) \\
&= \sum_{n=1}^N \ln [ \mu^{x_n} (1-\mu)^{(1 - x_n)} ] \\
&= \sum_{n=1}^N [x_n\ln\mu + (1-x_n)\ln(1-\mu)] \\
\end{split} 
$$


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
