---
layout: page
title: Sparse factorization of dense matrices
description: Fade-out instead of fill-in
img:
---

### Introduction
*This post summarizes joint work with [Tim](http://www.tjsullivan.org.uk/) and [Houman](http://users.cms.caltech.edu/~owhadi/index.htm) on the sparse Cholesky factorization of dense kernel matrices. In the interest of conciseness, I will defer to our [paper](https://arxiv.org/abs/1706.02205) for discussions of related work and technical details.*
*

Positive definite kernel matrices of the form

$$ \Theta_{ij}  \coloneqq \mathcal{G}\left(x_i, x_j\right)$$

play an important role in many parts of computational mathematics.
They arise as covariance matrices of Gaussian processes in statistics and as discretized solution operators of partial differential equations in computational engineering. 
By means of the [kernel trick](https://en.wikipedia.org/wiki/Kernel_method#Mathematics:_the_kernel_trick), they allow machine learning algorithms to employ infinite dimensional features maps.

This post focuses on covariance matrices of finitely smooth Gaussian processes or, equivalently, the solution operators of elliptic partial differential equations (PDEs).
Qualitatively, these kernels assign larger values to pairs of nearby points and smaller values to pairs of distant points.
This means that if we observe a smooth random process to be positive in a point $$x$$, we will strongly expect it to be positive at a nearby point $$y$$.
Based on this information, we will also tend to believe values at a more distant point $$z$$ to be positive, but we will be less confident in this belief.