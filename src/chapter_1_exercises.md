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
