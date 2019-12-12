---
layout: page
title: Competitive Gradient Descent
description: What is gradient descent for multi-player games?
img: /assets/img/oscillationSimGD.png
---

# Competitive Gradient Descent

## Introduction
*This post summarizes joint work with [Anima Anandkumar](http://tensorlab.cms.caltech.edu/users/anima/) on a new algorithm for competitive optimization: [Competitive gradient descent](https://nips.cc/Conferences/2019/ScheduleMultitrack?event=13843).*
    
Many learning algorithms are model a single agent minimizing a loss function, such as empirical risk.
However the spectacular successes of generative adversarial networks (GANs) have renewed interest in algorithms that are modeled after multiple agents that compete in optimizing their own objective functions, which we refer to as *competitive optimization*.

Much of single agent machine learning is powered by variants of gradient descent, which leads us to the important question:
**What is the natural generalization of gradient descent to competitive optimization?**


## Gradient Descent (GD)

Consider a single-agent optimization problem, $$ \min_{x \in \mathbb{R}^{m}} f(x) $$.
Gradient descent with step size $\eta$ is given by the update rule

$$ x_{k+1} = x_{k} - \eta \nabla f(x_{k}) $$

$$
   \begin{equation*}
  x_{k+1} = x_{k} - \eta \nabla f(x_{k})
  \end{equation*} 
$$

$$ \begin{equation*}
  x_{k+1} = x_{k} - \eta \nabla f(x_{k})
  \end{equation*} $$

$$ x_{k+1} = x_{k} - \eta \nabla f(x_{k}) $$ 

$$ x_{k+1} = x_{k} - \eta \nabla f(x_{k}) $$

$$ \begin{pmatrix} x_{k+1} \\ y_{k+1} \end{pmatrix} $$

$$ \begin{pmatrix}
  x_{k+1}\\
  y_{k+1}
\end{pmatrix} ≔
\begin{pmatrix}
  x_{k}\\
  y_{k}
\end{pmatrix} - \eta
\begin{pmatrix}
  I + \eta D^2_{xx}f & \eta D^2_{xy}f \\
  \eta D^2_{yx}g & I + \eta D^2_{yy}g
\end{pmatrix}^{-1}
\begin{pmatrix}
  \nabla_{x}f\\
  \nabla_{y}g
\end{pmatrix}. $$



where the gradient $$\nabla f(x_{k})$$ is the vector containing the partial derivatives of $$f$$, taken in the last iterate $$x_k$$.
The vector $$-\nabla f(x_{k})$$ points in the direction of the steepest descent of the loss function $$f$$ in the point $$x_k$$, which is why gradiend descent is also referred to as the method of steepest descent.

Let us now move to the competitive optimization problem:

$$
\min_{x \in \mathbb{R}^m} f(x, y) \newline 
\min_{y \in \mathbb{R}^n} g(x, y)
$$

restricting ourselves to two agents for the sake of simplicity.
Here, the first agent tries to choose $$x$$ such as to minimize $$f$$, while the second agent tries to choose the decision variable $$y$$ such as to minimize $$g$$.
The interesting part is that the optimal choice of $$x$$ depends of $$y$$ and vice versa, and the objectives of the two players will in general be at odds with each other, the important special case $$f = -g$$ corresponding to zero-sum or minimax games.

Since neither player can *know* what the other player will do, they might assume each other to not move at all.
Thenext move of their opponent, both agents might as well assume each other to be stationary.
Under this assumption, following the dire

$$
  x_{k+1} = x_{k} - \eta \nabla_x f(x_{k}, y_{k})\\
  y_{k+1} = y_{k} - \eta \nabla_y f(x_{k}, y_{k}).
$$

Here, $$\nabla_x f(x_{k}, y_{k}) \in \mathbb{R}^m$$ and $$\nabla_y f(x_{k}, y_{k}) \in \mathbb{R}^n$$ denote the gradient with respect to the variables $$x$$ and $$y$$, respectively.

Unfortunately, even on the most simple bilinear minimiax problem $$f(x,y) = x^{\top} y = - g(x,y)$$, SimGD fails to converge to the Nash equilibrium $$(0,0)$$.
Instead, its trajectories form ever larger cycles as the two players chase each other in strategy space.
The oscillatory behavior of SimGD is not restricted to this toy problem and a variety of corrections have been proposed in the literature.

![](https://i.imgur.com/CBqQEWT.png =300x200)
![](https://upload.wikimedia.org/wikipedia/commons/thumb/6/67/Rock-paper-scissors.svg/460px-Rock-paper-scissors.svg.png =230x220)

*Even for the simple bilinear problem $f(x,y) = -g(x,y) = xy$, simultaneous gradient descent cycles to infinity rather than converge towards the nash equilibrium $(0,0)$. This can be seen as the analogue of "ROCK! PAPER! SCISSOR ROCK ..." in the eponymous hand game (right image taken from [wikimedia](https://upload.wikimedia.org/wikipedia/commons/thumb/6/67/Rock-paper-scissors.svg))*