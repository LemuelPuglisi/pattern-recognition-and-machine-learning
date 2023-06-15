# Chapter 1 Exercises

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

### Solution to exercise 1.2

Given 

\\[
E(\bar w) = \frac12 \sum_{n=1}^N \left( 
x_n^M w_M + \dots + x_n w_1 + w_0 - t_n 
\right)^2 + \frac\lambda2 || \bar w ||^2 
\\]

Compute the partial derivative

\\[
\frac{\partial E}{\partial w_i} = \left[
\sum_{n=1}^N x_n^i \left(
x_n^M w_M + \dots + x_n w_1 + w_0 - t_n 
\right) \right] + \lambda w_i
\\]

Set the partial derivatives to zero to get $w$ that minimizes \\(E\\)

\\[
\frac{\partial E}{\partial w_i} = 
\sum_{n=1}^N x_n^i \left(
x_n^M w_M + \dots + x_n w_1 + w_0 - t_n 
\right) + \lambda w_i = 0
\\]

\\[
= \sum_{n=1}^N
x_n^{M+i} w_M + \dots + x_n^{1+i} w_1 + x_n^{0 + i}w_0 - x_n^{i} t_n + \lambda w_i = 0 
\\]

\\[
= \sum_{n=1}^N
x_n^{M+i} w_M + \dots + x_n^{1+i} w_1 + x_n^{0 + i}w_0 + \lambda w_i = \sum_{n=1}^N x_n^{i} t_n
\\]

\\[
= w_M \sum_{n=1}^N x_n^{M+i} + \dots + w_i (\lambda + \sum_{n=1}^N x_n^{2i}) + \dots +
 w_0 \sum_{n=1}^N x_n^{0 + i} = \sum_{n=1}^N x_n^{i} t_n
\\]

\\[
= \hat A_{iM} w_M + \dots + \hat A_{ij} w_j + \dots + \hat A_{i0} w_0  = T_i 
\\]

\\[ = \sum_{j=1}^M \hat A_{ij}w_j = T_i \\]

where 

\\[
\hat A_{ij} = \begin{cases}
\lambda + \sum_{n=1}^N x_n^{2i} & \text{if } i = j \\\\
\sum_{n=1}^N x_n^{j+i} & \text{otherwise} 
\end{cases}
\\]

or simply 

\\[ \hat A = \lambda I_M + A \\]

### Solution to exercise 1.3

The probability of selecting an apple is

\\[
\begin{split}
P(a) &= P(a, r) + P(a, b) + P(a, g) \\\\
&= P(a \mid r)P(r) + P(a \mid b)P(b) + P(a \mid g)P(g) \\\\
&= 3/10 \times 0.2 + \frac12 \times 0.2 + 3/10 \times 0.6 = 0.34
\end{split}
\\]

Following the Bayes theorem, the probability of the selected box being green given that the selected fruit is an orange is

\\[
P(g \mid o) = \frac{P(o \mid g) P(g)}{P(o)} = \frac{0.3 \times 0.6}{0.36} = 0.5
\\]

where 

\\[
\begin{split}
P(o) &= P(o, r) + P(o, b) + P(o, g) \\\\
&= P(o \mid r)P(r) + P(o \mid b)P(b) + P(o \mid g)P(g) \\\\
&= 4/10 \times 0.2 + \frac12 \times 0.2 + 3/10 \times 0.6 = 0.36 
\end{split}
\\]

### Solution to exercise 1.4

Revised solution from Bishop's solution manual.

Let $g$ be a non-linear change of variable $x = g(y)$, for probability density functions we know that 
$$
p_y(y) = p_x(g(y)) \cdot |g'(y)|
$$
Let $\hat x, \hat y$ be the maximum of $p_x, p_y$ densities respectively. Let $s = \text{sign}(g'(y)) \in \{-1, 1\}$ and re-write:
$$
p_y(y) = p_x(g(y)) \cdot sg'(y)
$$
Differentiate both sides:
$$
p'(y) = s p'_x(g(y)) [g'(y)]^2 + sp_x(g(y))g''(y)
$$
Suppose that $\hat x = g(\hat y)$, then 
$$
\begin{split}
p'(\hat y) &= s p'_x(g(\hat y)) [g'(\hat y)]^2 + sp_x(g(\hat y))g''(\hat y)\\
&= s p'_x(\hat x) [g'(\hat y)]^2 +  sp_x(\hat x)g''(\hat y) \\
&= s \cdot 0 \cdot [g'(\hat y)]^2 +  sp_x(\hat x)g''(\hat y) \\
&= sp_x(\hat x)g''(\hat y) = 0 \\

\end{split}
$$
Where:

1. $s \in \{-1, 1\}$ cannot be zero
2. $p_x(\hat x)$ is the maximum probability, thus cannot be zero

This means $\frac{\partial^2 g(\hat y)}{\partial y^2}$ has to be 0, which depends on $g$, hence the relation $\hat x = g(\hat y)$ may not hold. If $g$ is linear, then the second derivative of $g$ is 0 and the relation $\hat x = g(\hat y)$ is valid.

### Solution to exercise 1.5

$$
\begin{split}
var[X] &= \int p(x) \bigg[ f(x) - \mathbb{E}[f(x)] \bigg]^2 dx \\
&= \int p(x) \bigg[  f(x)^2 -2 \mathbb{E}[f(x)] f(x) + \mathbb{E}[f(x)]^2 \bigg]dx\\
&= \int (p(x)f(x)^2 - 2p(x)\mathbb{E}[f(x)] f(x) + p(x)\mathbb{E}[f(x)]^2) dx\\
&= \int p(x)f(x)^2 dx - 2\int p(x)\mathbb{E}[f(x)]f(x)dx + \int p(x)\mathbb{E}[f(x)]^2 dx\\
&= \mathbb{E}[f(x)^2] - 2\mathbb{E}[f(x)] \int p(x)f(x)dx + \mathbb{E}[f(x)]^2 \int p(x)dx \\
&= \mathbb{E}[f(x)^2] - 2\mathbb{E}[f(x)]^2 + \mathbb{E}[f(x)]^2 \\
&= \mathbb{E}[f(x)^2] - \mathbb{E}[f(x)]^2
\end{split}
$$

### Solution to exercise 1.6

From 1.41

$$
cov(x,y) = \mathbb{E}_{x,y}[xy] - \mathbb E[x]\mathbb E[y]
$$

But if x and y are indipendent, then

$$
\begin{split}
\mathbb{E}_{x,y}[xy] &= \int\int p(x,y)xy\space dxdy \\
&=  \int\int xp(x)\space yp(y) \space dxdy \\
&= \mathbb{E}[x] \int \space y p(y) dy \\
&= \mathbb{E}[x]\mathbb{E}[y]
\end{split}
$$

Therefore $cov(x,y)=0$.

### Solution to exercise 1.32

Let $x \sim p_x(x)$ and let $y = Ax$ be a linear change of variable. In that case, the jacobian factor is the determinant $|A|$ and we can write

$$
p_y(y) = p_x(y) |A|
$$

So we can write

$$
\begin{split}
H(y) &= - \int p_y(y) \ln p_y(y) dy \\
&= - \int p_y(y) \ln [p_x(y) |A|] dy \\
&= -\int p_y(y)\left[ \ln p_x(y) + \ln |A| \right] dy \\
&= \bigg[-\int p_y(y)\ln p_x(y) dy \bigg] - \ln |A|\int p_y(y)dy \\
&= \bigg[-\int p_y(y)\ln p_x(y) dy \bigg] - \ln |A|
\end{split}
$$

(last steps: solve the integral on the left-hand side using the substitution $A^{-1}y=x$ remembering that A is non-singular)

