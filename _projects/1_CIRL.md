---
layout: page
title: Constrained IRL.
description: Finding provable convergence guarantees for constrained inverse reinforcement learning (CIRL).
img: assets/img/publication_preview/dual_descent.gif
importance: 1
category: work
related_publications:
---
<!-- 
Inverse reinforcement learning (CIRL) describes a class of algorithms that learn a reward function "motivating" a behavior from a dataset of expert demonstrations. While ensuring that a set of pre-defined constraints are met.  -->

During my Master thesis at EPFL I proposed an algorithm called `NPG-CIRL` which solves the problem of constrained inverse reinforcement learning (CIRL) and showed convergence guarantees. 
The `CIRL` problem is an extension of the inverse reinforcement learning problem (`IRL`), which is concerned with recovering a reward function explaining the behavior of an expert agent. 
IRL methods are generally considered part of the broader class of imitation learning methods, which aim to enable agents to learn behaviors from demonstrations. 
Such approaches have seen great success in the field of robotics and has, for instance, [been used to train locomotive policies in robotic dogs](https://doi.org/10.15607/RSS.2020.XVI.064), using expert data acquired by motion capture on real canines.
Rewards recovered through IRL methods can, in turn, be used to clone the behavior of that expert. 

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/IRL_transfer.gif" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    An illustration of how the transferability property in IRL.
</div>

Compared to alternative methods, such as behavioral cloning that directly tries to reproduce an expert policy with supervised learning, `IRL` presents the advantage of recovering a representation of the underlying goal of the expert. Depending on the problem structure, especially in MDPs with sparse rewards, the reward may provide a more compressed representation of the goal of an agent than the expert policy. 
Furthermore, this representation describes a goal independent of the underlying MDP dynamics. That makes IRL better suited to learn policies that are transferable across different dynamics, making IRL methods particularly adapted to learning behaviors that generalize across different settings. 


`CIRL` differs from the more extensively studied `IRL` problem by introducing known constraints into the problem formulation. The introduction of constraints guarantees that the recovered reward induces a policy that meets explicitly specified requirements. This property is crucial in safety-critical domains, such as autonomous vehicles, robotics, or medical applications. 
Another advantage presented when introducing known constraints into IRL is that `CIRL` recovers a reward that does not implicitly represent the constraints.
**`CIRL` learns a representation that decouples the reward from the constraints.**
That property implies, for instance, that a self-driving model trained on a dataset of Swiss roads could be used to learn a reward function which does not represent known constraints such as speed limits and that, when specified different restrictions, it could still meet them. For instance, the model would not require a new training dataset to learn to drive faster than the Swiss speed limitations when on the German autobahn. This would not be achievable with an unconstrained IRL algorithm.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/CIRL_speed.gif" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    An illustration of how CIRL learns rewards that are decoupled from constraints.
</div>


The main contribution of my thesis was to introduce, `NPG-CIRL`, a primal dual scheme which extends the well-studied [natural policy gradient](http://papers.neurips.cc/paper/2073-a-natural-policy-gradient.pdf) (`NPG`) algorithm and is similar to methods presented to solve the `IRL`. `NPG-CIRL` differs from the CRL algorithm of by the introduction of entropy regularization and by the fact that two dual variables are studied, with one of them being parametrized (in this work, we consider linearly parametrized rewards). 


The main contribution of the thesis lies in providing a detailed analysis of the `NPG-CIRL` method.
1. We prove that in the exact gradient setting, under softmax policy parameterization, and linear reward parameterization, our method globally converges at a $$ O(1/\sqrt{T})$$ rate.
2. We study the convergence of our algorithm in the stochastic setting, where gradients are estimated using Monte Carlo estimators. We show that, assuming the global convergence rate of $$O(1/\sqrt{T})$$ still holds, even when the gradient estimators are biased.

## The algorithm and a sketch of convergence


<!-- 
{% raw %}
```html
<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.html path="assets/img/6.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm-4 mt-3 mt-md-0">
        {% include figure.html path="assets/img/11.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
```
{% endraw %} -->
