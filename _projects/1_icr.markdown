---
layout: page
title: Implicit Competitive Regularization
description: How do GANs generate?
img: /assets/gif/icr_title.gif
---

### Minimax is not enough

*This post summarizes joint work with [Anima](http://tensorlab.cms.caltech.edu/users/anima/) and [Hongkai](https://devzhk.github.io/). In the interest of conciseness, I will defer to our [paper](https://arxiv.org/abs/1910.05852) for discussions of related work and technical details.*

[GANs](https://en.wikipedia.org/wiki/Generative_adversarial_network) are fascinating!
Not only can they generate strikingly realistic [images](https://github.com/NVlabs/stylegan), they also introduce an exciting new paradigm to mainstream machine learning.

Where ordinary neural networks learn by minimizing a fixed loss function, GANs consist of two neural networks that compete with each other in a zero-sum game. A generator produces fake images while a discriminator tries to distinguish them from real ones.

In most work on GANs, this is seen as the generator minimizing just another loss function that happens to be obtained by fully optimizing the discriminator.

I will argue that **this minimax interpretation of GANs can not explain GAN performance**.
Instead, GAN performance can only be explained by the *dynamics* of simultaneous training.

In an attempt to make this more precise, I will explain how *implicit competitive regularization* could allow GANs to generate good images.
I will also provide empirical evidence that this is what actually happens in practice.

### The GAN-dilemma

The objective function of the [original GAN](https://arxiv.org/abs/1406.2661) is

$$
  \min \limits_{\mathcal{G}} \max \limits_{\mathcal{D}} \mathbb{E}_{x \sim P_{\mathrm{data}}}\left[\log \left(\mathcal{D}(x)\right)\right] + \mathbb{E}_{z \sim \mathcal{N}}\left[\log \left( 1 - \mathcal{D}\left( \mathcal{G}\left(z\right)\right)\right)\right].
$$

Here, $$x \sim P_{\mathrm{data}}$$ is sampled from the training data and $$z \sim \mathcal{N}$$ from a multivariate normal.
The generator network $$\mathcal{G}$$ learns to map $$z$$ to fake images and the discriminator network $$\mathcal{D}$$ learns to classify images as real or fake.

If we take the maximum over all possible functions $$\mathcal{D}$$, we obtain the [Jensen-Shannon divergence](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence) (JSD) between the distributions of real and fake images.
Therefore, it is believed that GANs work by minimizing the JSD between real and generated data.
Subsequently, other GAN variants were proposed that are modeled after different divergences or metrics between probability distributions.

However, any such interpretation runs into one of the two following problems.

+ Without regularity constraints, the discriminator can (almost) always achieve perfect performance

+ Imposing regularity constraints needs a measure of similarity of images, which is hard to obtain.

We call this the **GAN-dilemma**.

#### Unconstrained discriminators can always be perfect

The original GAN falls into the first category. For instance, if we have a finite amount of training data and the generated distribution has a density, the discriminator can assign arbitrarily high values to the real data points, while assigning arbitrarily low values anywhere else.

Thus, for an optimal discriminator, the generator loss always has the same value.
Therefore, it can not measure the relative quality of different generators.

<div class="img_row">
    <img class="col threehalf left" src="{{ site.baseurl }}/assets/gif/not_picking_out.gif" alt="" title="What we would like the discriminator to do"/>
    <img class="col threehalf left" src="{{ site.baseurl }}/assets/gif/picking_out.gif" alt="" title="What it might actually do"/>
</div>
<div class="col three caption">
  We would like the discriminator to compare the local density of true and fake datapoints (left). But without constraints, it can just pick out individual datapoints to achieve arbitrarily low loss, without providing a meaningful assessment of the generator's quality.
</div>


#### Measuring visual similarity is hard!

This observation led to the development of [WGAN](https://arxiv.org/abs/1701.04862), which instead uses (approximately) the formulation

$$
  \min \limits_{\mathcal{G}} \max \limits_{\mathcal{D}\colon \|\nabla \mathcal{D}\| \leq 1} \mathbb{E}_{x \sim P_{\mathrm{data}}}\left[\mathcal{D}(x)\right] - \mathbb{E}_{z \sim \mathcal{N}}\left[ \mathcal{D}\left( \mathcal{G}\left(z\right)\right)\right].
$$

The key difference here is the constraint on the discriminator.
WGAN restricts the size of the gradient of the Discriminator by one, forcing it to map nearby points to similar values.
Thus, the generator loss under an optimal discriminator will be smaller if the generated images are closer to the true images. In fact, it will be equal to the [earth mover's distance](https://en.wikipedia.org/wiki/Earth_mover%27s_distance) between the two distributions.

The big catch is that we have to choose a way to quantify the size $$\left\| \nabla \mathcal{D}\right\|$$ of the discriminator's gradients!
Since the inputs of the discriminator are images, this  means we have to measure similarity of images.

Most variants of WGAN bound the Euclidean norm $$\left\| \nabla \mathcal{D}\right\|_2$$ of the discriminator's gradient.
But this amounts to measuring the similarity images by the Euclidean distance of the respective vectors of pixel-wise intensities.
The example below shows that this is a *terrible* measure of visual similarity.

<div class="img_row">
    <img class="col three center" src="{{ site.baseurl }}/assets/img/deception_resize.png" alt="" title="The images are ordered from left to right in increasing order of distance. The first pair of images is identical, while the third pair of images differs by a tiny warping."/>
</div>
<div class="col three caption">
    Above you see three pairs of images (left column, middle column, and right column). Can you guess the ranking of the pixel-wise Euclidean distance within each pair? Hover over the image to see the solution. 
</div>

In general, **quantifying visual similarity between images is a longstanding open problem in computer vision.**
Until this problem is solved we will not be able to meaningfully constrain the discriminator's regularity.

### A way out

In the above, I am not arguing that GANs cannot work. They work remarkably well!
I am arguing that the minimax interpretation is but a red herring that has nothing to do with GAN performance.

Generative modeling is all about generating data *similar* to the training data.
Since GANs can create realistic images, they must have access to a notion of image similarity that captures visual similarity. 
Most GAN variants can achieve good results, so this does not seem to be the result of a particular choice of loss function.
Instead, it has to arise from the [inductive biases](https://en.wikipedia.org/wiki/Inductive_biasl) of the neural network that parametrizes the discriminator.

Deep neural networks reliably learn patterns obvious to the human eye.
This suggests that they capture *something* about visual similarity better than the feature maps, kernels, or metrics of classical computer vision.\\
The problem is that we cannot just *open up the network* to access this notion of similarity. Instead, it only arises implicitly, through the training process.
In particular, the output of a neural network classifier on a sample does [not reflect the uncertainty](https://arxiv.org/abs/1706.04599) of the classification.
This means that all information about how similar the fake images look to the real ones is *lost* once the discriminator is fully trained.

I think that the magic of GANs has to lie instead in the *dynamics* of simultaneous training that allows us to use the inductive biases of the discriminator for image generation, *without explicitly characterizing them*.
I will now attempt to explain *how* this could be happening.

#### Implicit competitive regularization (ICR)

Simultaneous gradient descent (SimGD) has stable points that are *unstable* when only training one of the players with gradient descent, while keeping the other player fixed.
We call this phenomenon *implicit competitive regularization* (ICR).
For instance, we can consider the quadratic problem

$$
\min \limits_x \max \limits_y x^2 + 10 xy + y^2
$$

and observe that SimGD with step sizes $$\eta_x = 0.09 , \eta_y = 0.01$$ converges to $$(0,0)$$ even though this is the *worst* choice for the maximizing player.
If we instead keep $$x$$ fixed by setting $$\eta_x = 0$$ and train $$y$$ using gradient descent, it will diverge to infinity for almost all starting points.

<div class="img_row">
    <img class="col three center" src="{{ site.baseurl }}/assets/gif/combined_stable.gif" alt="" title="When training ."/>
</div>
<div class="col three caption">
    Above you see three pairs of images (left column, middle column, and right column). Can you guess the ranking of the pixel-wise Euclidean distance within each pair? Hover over the image to see the solution. (If the image is cropped, open it in a separate tab.)
</div>

This is commonly seen as a *flaw* of SimGD, but I think it is crucial for GANs to work.
Just like $$y$$ can improve for any fixed $$x$$, a GAN discriminator can improve for any fixed generator.
Therefore, our only hope for convergent behavior in GANs is ICR! 

To verify this behavior in the wild, we train a GAN on MNIST until training stagnates and the generator produces good images. We then either train only the discriminator using gradient descent (keeping the generator fixed), or continue training both players using SimGD.
We observe that the discriminator changes more rapidly when trained in isolation, suggesting that the point of departure was indeed stabilized by ICR.

<div class="img_row">
    <img class="col threehalf left" src="{{ site.baseurl }}/assets/img/loss_compare_resize.png" alt="" title="Discriminator loss keeps decreasing when only training discriminator"/>
    <img class="col threehalf right" src="{{ site.baseurl }}/assets/img/pred_D_resize.png" alt="" title="Under simultaneous training, the discriminator changes slowly"/>
</div>
<div class="col three caption">
    We train a GAN on MNIST until we reach a good checkpoint. We then train only the discriminator, or both networks simultaneously.
    On the left we see that when training only the discriminator, its loss drops to near zero in accordance to the first part of the GAN-dilemma. 
    On the right we see how the discriminator output on 500 real and 500 fake test images compares to that of the checkpoint discriminator.
    When training simultaneously, the discriminator changes very slowly. An evidence for ICR!
</div>


#### ICR selects for slowly learning discriminators

You might have noticed the peculiar choice of learning rates for the two player in the above example.
This was not due coincidence, but due to a fascinating property of ICR: *It selects for the relative speed of learning of $$x$$ and $$y$$!*.

If we play around with the learning rate a bit more we notice that ICR is stronger if $$y$$ (the player that would want to run off to infinity) learns slowly compared to $$x$$.

<div class="img_row">
    <img class="col one left" src="{{ site.baseurl }}/assets/gif/beta_1.gif" alt="" title="When training ."/>
    <img class="col one center" src="{{ site.baseurl }}/assets/gif/beta_0.gif" alt="" title="When training ."/>
    <img class="col one right" src="{{ site.baseurl }}/assets/gif/beta_-1.gif" alt="" title="When training ."/>
</div>
<div class="col three caption">
  We have $$\eta_x = 0.09$$,
</div>


#### What ICR has to do with image quality

In the quadratic example above, the gradient of the loss only vanishes in $$(0,0)$$ and ICR determined whether the SimGD converges to this point.
A real GAN is highly nonlinear and has many points with (almost) vanishing gradient.
ICR then determines to which, if any, of these points SimGD converges.

Since training and generalization of deep neural networks is poorly understood even for classification problems, we will need to make an assumption regarding the relationship of discriminator training and the quality of the samples generated by the generator:

**Hypothesis:** The speed with which the discriminator picks up on an imperfection of the generator measures the (human) visual prominence of the imperfection.
Imperfections that are visually obvious will be picked up on more quickly than those that are visually subtle.

We have not yet verified this hypothesis rigorously, but it expresses the intuition that neural networks first learn simpler and more general patterns, before learning more intricate, specific ones.
In this sense, it is also in line with the recently proposed [coherent gradient hypothesis](https://openreview.net/forum?id=ryeFY0EFwS).

As we have seen in the last section, ICR preferably stabilizes pairs of points where the discriminator learns slowly. By our hypothesis, these correspond to generators that are accurate in the most visually important directions, and that therefore produce good samples.
Thus, ICR identifies good generators without requiring an explicit measure of images similarity.

### Improving GAN training by strengthening ICR

So how does the above help us to improve GANs?
One problem with GANs is that their training is often not unstable.
In fact, training them for too long can even deteriorate the resulting images.
This fits nicely with what we observed so far.
ICR stabilizes good generators to some degree, but it is often not strong enough to lead to convergence.
Instead, it stabilizes the system only temporarily.
How can we strengthen ICR to better stabilize these points?

#### A game perspective on ICR

Consider again the example

$$
\min \limits_x \max \limits_y x^2 + xy + 10^{-1} y^2
$$

where both players converge to zero, when using SimGD.
Why do greedy updates from both players let $$y$$ converge to its *worst* strategy?

The answer is that the other strategies are vulnerable to *counterattack* by $$x$$. 
Indeed, if $$y$$ were to move to $$\epsilon$$, $$x$$ would move start moving towards $$-2 \epsilon$$, using the mixed term $$xy$$ to improve its loss.
The loss of $$y$$ due to the mixed term will outweigh the gains due to the quadratic terms, prompting $$y$$ to move back towards zero.

#### Competitive gradient descent for stronger ICR

In SimGD, $$y$$ only takes the reaction of $$x$$ into account after it has occurred.
This is the reason why SimGD does not converge to zero in the bilinear problem 

$$\min \limits_x \max \limits_y xy$$

 and more generally induces relatively weak ICR.
[Competitive Gradient Descent (CGD)](https://f-t-s.github.io/projects/cgd/) lets both players try to anticipate each other's action, at every iteration of the algorithm.
This greatly increases the effect of ICR.

When applying CGD to the quadratic example, we see that it converges over a much larger range of step sizes.

When applying CGD to the example on MNIST, we see that *overtraining* the discriminator using CGD will make it even more robust to counteraction of the discriminator.

When applying CGD to the model problem computing the $$\eta$$-projection, we observe that it greatly prolongs the stability of the projection state.

#### Experiments on CIFAR 10

Based on the above, we hoped that strengthening ICR by training GANs with CGD would lead to better results than explicit regularization through, for instance, gradient penalties.
To this end, we used the same DCGAN-architecture as in [WGAN-GP](https://arxiv.org/abs/1704.00028) and combined it with a wide range of regularizers and loss function.
Indeed, we find that training with an adaptive version of CGD yields better, and more consistent results, as measured by the inception score.
We see this as additional support that ICR is a key element of GAN performance.

### Conclusion

Instead of scrambling to salvage the minimax point of view, I think that it is more practical and more interesting to *embrace* the fact that these algorithms are designed as an iterative game and *do not* amount to a single player implicitly minimizing a loss function.\\
This requires us to better understand why points found by adversarial training should be useful for a given downstream task.

I hope that the proposal of ICR as underlying mechanism will pave the way for more sophisticated understanding of adversarial training.
In the long run, I believe that this will greatly the range of problems that can be solved using deep neural networks.
