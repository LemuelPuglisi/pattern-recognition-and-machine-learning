
## The Evidence Approximation

Our linear regression model currently depends on the weights $w$ and on the hyperparameters $\alpha, \beta$ (see prev. paragraphs). A fully Bayesian treatment would introduce prior distribution over all the parameters and hyperparameters, and calculate the predictive distribution by marginalization. Anyway, solving the integral of the marginalization on all these parameters is analytically intractable.

If we introduce two priors over $\alpha, \beta$ (hyperpriors), then the predictive distribution is obtained by marginalizing over $w, \alpha, \beta$ as follows:

$$
p(t \mid T) \int\int\int p(t \mid w,\beta) p(w\mid T, \alpha, \beta) p(\alpha, \beta \mid T) \space dw\space d\alpha\space d\beta
$$

Where $p(t \mid w, \beta)$ is a likelihood function (given by 3.8)$ and $p(w \mid T, \alpha, \beta)$ is the posterior (the Gaussian with mean $m_N$ and covariance matrix $S_N$) and $p(\alpha, \beta \mid T)$ is a posterior for the hyperparameters.

An approximation, called **Empirical Bayes**, is given by:
1. Obtaining the marginal likelihood by integrating over $w$
2. Maximizing the likelihood to obtain the hyperparameters

Another approximation can be used if the posterior $p(\alpha, \beta \mid T)$ is peaked around the values $\hat \alpha, \hat \beta$. In this case we just obtain the two values, replace them in the marginalization, and we marginalize over $w$:

$$
p(t \mid T) \approx p(t \mid T, \hat \alpha, \hat \beta) = \int p(t \mid w, \hat \beta) p(w \mid T, \hat \alpha, \hat \beta) \space dw
$$

From Bayes theorem we know that:

$$
p(\alpha, \beta \mid T) \propto p(T \mid \alpha, \beta) p(\alpha, \beta)
$$

If the prior $p(\alpha, \beta)$ is relatively flat, then $\hat \alpha, \hat \beta$ can be obtained by maximizing the likelihood $p(T \mid \alpha, \beta)$ instead of the posterior $p(\alpha, \beta \mid T)$. 


But how do we compute the likelihood $p(T \mid \alpha, \beta)$? Let's marginalize over $w$:

$$
\begin{split}
p(T \mid \alpha, \beta) &= \int p(T \mid w, \beta)p(w \mid \alpha) \space dw\\
\vdots \\
&= \left(\frac{\beta}{2\pi}\right)^{N/2} \left(\frac{\alpha}{2\pi}\right)^{M/2} \int \exp \{ -E(w) \} \space dw
\end{split}
$$

where 

$$
\begin{split}
E(w) &= \beta E_D(w) + \alpha E_W(w) \\
&= \frac{\beta}{2} ||T - \Phi w ||^2 + \frac\alpha2 w^Tw
\end{split}
$$

If you want to know the intermediate calculation denoted by $\vdots$, read the content of this image:

![calcs](./assets_ch3/derivation_of_3-78.png)








