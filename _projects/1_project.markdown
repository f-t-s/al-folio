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

Consider a single-agent optimization problem,
$$
  \min_{x \in \mathbb{R}^{m}} f(x).
$$
Gradient descent with step size $\eta$ is given by the update rule

$$
  x_{k+1} = x_{k} - \eta \nabla f(x_{k})
$$

where the gradient $$\nabla f(x_{k})$$ is the vector containing the partial derivatives of $$f$$, taken in the last iterate $$x_k$$.
The vector $$-\nabla f(x_{k})$$ points in the direction of the steepest descent of the loss function $$f$$ in the point $$x_k$$, which is why gradiend descent is also referred to as the method of steepest descent.

Let us now move to the competitive optimization problem:

$$
\min_{x \in \mathbb{R}^m} f(x, y)\\
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




Every project has a beautiful feature shocase page. It's easy to include images, in a flexible 3-column grid format. Make your photos 1/3, 2/3, or full width.

To give your project a background in the portfolio page, just add the img tag to the front matter like so:

    ---
    layout: page
    title: Project
    description: a project with a background image
    img: /assets/img/12.jpg
    ---


<div class="img_row">
    <img class="col one left" src="{{ site.baseurl }}/assets/img/1.jpg" alt="" title="example image"/>
    <img class="col one left" src="{{ site.baseurl }}/assets/img/2.jpg" alt="" title="example image"/>
    <img class="col one left" src="{{ site.baseurl }}/assets/img/3.jpg" alt="" title="example image"/>
</div>
<div class="col three caption">
    Caption photos easily. On the left, a road goes through a tunnel. Middle, leaves artistically fall in a hipster photoshoot. Right, in another hipster photoshoot, a lumberjack grasps a handful of pine needles.
</div>
<div class="img_row">
    <img class="col three left" src="{{ site.baseurl }}/assets/img/5.jpg" alt="" title="example image"/>
</div>
<div class="col three caption">
    This image can also have a caption. It's like magic.
</div>

You can also put regular text between your rows of images. Say you wanted to write a little bit about your project before you posted the rest of the images. You describe how you toiled, sweated, *bled* for your project, and then.... you reveal it's glory in the next row of images.


<div class="img_row">
    <img class="col two left" src="{{ site.baseurl }}/assets/img/6.jpg" alt="" title="example image"/>
    <img class="col one left" src="{{ site.baseurl }}/assets/img/11.jpg" alt="" title="example image"/>
</div>
<div class="col three caption">
    You can also have artistically styled 2/3 + 1/3 images, like these.
</div>


<br/><br/>


The code is simple. Just add a col class to your image, and another class specifying the width: one, two, or three columns wide. Here's the code for the last row of images above:

<div class="img_row">
    <img class="col two left" src="/img/6.jpg"/>
    <img class="col one left" src="/img/11.jpg"/>
</div>
