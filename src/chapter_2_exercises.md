# Chapter 2 Exercises

### Solution to exercise 2.1

Prop n.1.

$$
\sum_{x=0}^1 p(x\mid\mu) = \mu + (1-\mu) = 1  
$$

Prop n.2

$$
\mathbb E [x] = \sum_{x=0}^1 x p(x\mid \mu) =
\sum_{x=0}^1 x \cdot \mu^x (1-\mu)^{1-x} =
1 \cdot \mu^1 (1-\mu)^{1-1} = \mu
$$

Prop n.3

$$
\begin{split}
var[x] &= \sum_{x=0}^1 p(x)(x - \mu)^2 \\
&= p(0)\mu^2 + p(1)(1-\mu)^2 \\ 
&= (1-\mu)\mu^2 + \mu(1-\mu)^2 \\
&= \mu((1-\mu)\mu + (1-\mu)^2) \\
&= \mu(\mu - \mu^2 + 1 - 2\mu + \mu^2) \\
&= \mu(1-\mu)
\end{split} 
$$


### Solution to exercise 2.30

We know that

$$
\mathbb{E}[z] = R^{-1} \begin{bmatrix}
\Lambda \mu - A^T Lb \\
Lb
\end{bmatrix}
$$

and that 

$$
cov[z] = R^{-1} = \begin{bmatrix}
\Lambda^{-1} & \Lambda^{-1}A^T \\
A\Lambda^{-1} & L^{-1}+ A \Lambda^{-1}A^T 
\end{bmatrix}
$$

So by replacing $R^{-1}$ we get

$$
\mathbb{E}[z] = 
\begin{bmatrix}
\Lambda^{-1} & \Lambda^{-1}A^T \\
A\Lambda^{-1} & L^{-1}+ A \Lambda^{-1}A^T 
\end{bmatrix}
\begin{bmatrix}
\Lambda \mu - A^T Lb \\
Lb
\end{bmatrix}
$$

which results in 

$$
\begin{bmatrix}
\mu  - \Lambda^{-1}A^TLb + \Lambda^{-1}A^TLb \\
A\mu - A \Lambda^{-1}A^TLb + b + A \Lambda^{-1}A^TLb
\end{bmatrix}
=
\begin{bmatrix}
\mu \\
A\mu + b
\end{bmatrix}
$$



### Solution to exercise 2.31

There are various approaches to compute the marginal distribution $p(y)$ where $y = x + z$ and $x \sim \mathcal{N}(x \mid \mu_x, \Sigma_x)$, $z \sim \mathcal{N}(z \mid \mu_z, \Sigma_z)$. 

The first approach came to my mind after a video from [3Blue1Brown](https://www.youtube.com/watch?v=IaSGqQa5O-M), that demonstrates exactly that in this case $p(y) = p(x) * p(z)$, where $*$ is the convolution operator.

$$
p(y) = \int p_x(y-t)p_z(t)dt
$$

In this way, we consider every way to obtain $y$ from the sum $x+z$. This solution has been adopted by Tommy Odland in his [solutions](https://tommyodland.com/files/edu/bishop_solutions.pdf). 

But there is a simpler way to do this. Let's consider the conditional distribution $p(y \mid x)$. Since $x$ is fixed, and $y=x+z$, the only variability is up to $z$. We can define this as

$$
p(y\mid x) = \mathcal N (y \mid \mu_z + x, \Sigma_z)
$$

We can now compare the obtained results with expressions 2.99 and 2.100:

$$
\mu = \mu_x \hspace{.5cm}
\Lambda^{-1} = \Sigma_x \hspace{.5cm}
A = I \hspace{.5cm}
b = \mu_z \hspace{.5cm} 
L^{-1} = \Sigma_z \hspace{.5cm}
x=x
$$

And using results from 2.109 and 2.110 we know:

$$
\mathbb{E}[y] = A\mu+b = I \mu_x + \mu_z = \mu_x + \mu_z
$$

and

$$
cov[y] = L^{-1} + A\Lambda^{-1} A^T = \Sigma_z + \Sigma_x
$$
