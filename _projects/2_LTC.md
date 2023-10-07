---
layout: page
title: RL with Time Continuous Neural Nets
description: Learning motor policies with time continuous neural networks.
img: assets/img/LTC.jpg
importance: 2
category: work
---

Time-Continuous Neural Networks provide an effective framework for the modeling of dynamical systems and are natural candidates for continuous-control tasks. They are closely related to the dynamics of non-spiking neurons, which gives further justification to investigate their use in control. This post gives a summary of the work and results obtained while investigating the use of such neural networks during a semester project that I worked jointly supervised by EPFL’s [BIOROB](https://www.epfl.ch/labs/biorob/) and [LCN](https://www.epfl.ch/labs/lcn/) labs.

## Time-continuous neural networks

Time-Continuous Neural Networks model dynamical systems model the evolution of the hidden states (which we denote as $$x_t$$ at a given time $$t$$) of a neural network by equations of the form:


$$
\frac{\partial x_t}{\partial t} = D(xt, It, θ)
$$


Where D denotes some kind of model function that estimates the time-derivative of $$x_t$$. $$D$$ is a function of $$x_t$$, which denotes the hidden state of the neuron, It which denotes the inputs of the neuron and a learnable parameter vector $$\theta$$. Updates of the (hidden) state $$x_t$$ are the computed using some ODE solver which integrates the flow $$D(x_t, I_t, \theta)$$ over some time-step $$\Delta_t$$ to compute: 

$$
x_{t+\Delta_t} = \int^{t+\Delta_t}_t D(x_t, I_t, \theta).
$$

It is worth noting that in a supervised learning context, this formulation has the advantage of being able to represent irregularly sampled time-sequences. For the control applications we consider here, the sample-rate is imposed by the hardware of our robot and is likely regular.  

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/unroll.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Such a model can be though of as a recurrent neural network where the recurrent connection are implemented by an integrator.
</div>

The most straight forward approach to implementing a continuous-time neural network is to directly use a neural network to model the flow D this approach is often referred to as "[Neural-ODEs](https://arxiv.org/abs/1806.07366)":
Where the flow $$D(x_t, I_t, \theta)$$ is given by the output of a neural network:

$$
D(x_t, I_t, \theta) = f(x_t, I_t, θ).
$$

An alternative provided by an earlier contribution is the so called "[Continuous-Time Recurrent Neural Network](https://www.sciencedirect.com/science/article/abs/pii/S089360800580125X)" model (CT-RNNs) introduces a stabilization term to an equivalent formulation:

$$
D(x,I,\theta)=−\frac{x_t}{\tau} + f(x,I,\theta).
$$

Finally, a more recent approach referred to as "[Liquid Time Constant Neural Networks](https://arxiv.org/abs/2006.04439)" introduces further non-linearity by having $$f$$ affect the time-constant (hence make the time-constant "liquid"), this approach corresponds to the following time-derivative model:

$$
D(x,I,\theta)= - \Bigg[\frac{1}{\tau} +  A f(x,I,\theta) \Bigg]x_t +  f(x,I,\theta).
$$

Compared to multi-layer perceptrons and RNNs computing gradients on continuous time neural networks is less obvious because of the integrator step introduced in the computation of the inner state x. There are two different possible approach to the computation of such gradients together with their pros and cons: backpropagation through time and the adjoint sensitivity method.

Backpropagation through time (`BPTT`) works by directly computed the gradient of a loss function through the ODE solver (it requires our ODE solver to build a computation graph and then we use autograd to compute a gradient). `BPTT` requires saving the inner state of neurons for every step of the ODE solver, this comes at a high memory cost.


The adjoint sensitivity method, required saving a so-called adjoint state $$a(t) = \frac{\partial L}{\partial x_t}$$ throughout the unrolling of the neural network. Given some Loss function:

$$
L(x(t)) = L \Bigg( x(t_0) + \int_{t_0}^{t_1} D(xt, It, θ) dt\Bigg)
$$

we define our adjoint sensitivity state  $$a(t)$$. The idea here is to use the stored  $$a(t)$$ to compute a gradient:

$$
\frac{\partial L(x(t))}{\partial t} = \int_{t_1}^{t_0} a(t)^T \frac{\partial D(x(t),I_t,\theta)}{\partial \theta} dt
$$

Note that we compute the loss backwards through time (from $$t_1$$ to $$t_0$$) which is why the negative sign appears in front of the integral. Which we can use to perform gradient descent on our model. 
This computation is performed using the same numerical ODE solver as for the forward pass.

## Building a neural network cell for motor control

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/LTC.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    A visual representation of a fully connected Time-Continuous Neural Network Cell. With 4 inputs, 5 inner-neurons and 2 motor neurons.
</div>


In order to deal with the motor task that we consider in this project we choose to investigate small fully connected neuron cells that take a $$n$$-dimensional input, contain $$k$$ inner-neurons and return $$d$$ outputs (we call the outputs "motor neurons"). Most of our experiments are performed with LTC cells, so we will explicitly derive the forward pass equations for an LTC, but we can build equivalent cells for RNN-ODE and CT-RNN cells in a very similar fashion.

## RL training and experiments

In order to train our fully connected time-continuous neural network cell, we implement a batched version of the PPO algorithm, that we modify to work with the adjunct method of gradients computation.  
The reinforcement learning-based training of a time-continuous neuron cell requires the setup of a learning environment. Such an environment must be able to:
1. simulate the POMDP (since we investigate a continuous control task, this is the physics of our controlled system),
2. implement the policy model (in our case our time-continuous neuron cells),
3. train it using our RL algorithm (PPO).


<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/Framework.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Reinforcement learning environment architecture.
</div>


In order to investigate the efficiency of TCNN cells we setup a simple (and rather classical in the RL literature) motor control experiment: the cart-pole problem. We use the Mujoco gym implementation of cart-pole together with our implementation of `PPO` training for `TCNN` policies to perform our experiments. We use the adjoint method for the computation of gradients and focus our experiments on LTC cells. The code used for the experiments is available at [https://github.com/RenardDesNeiges/CTNN_Policies_DERL](https://github.com/RenardDesNeiges/CTNN_Policies_DERL).


<div class="row center">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/inv_pendulum_ltc_1.gif" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    An example trajectory produced by the trained policy.
</div>


We find that the LTC cell trained with PPO can find efficient solutions to the cart-pole problem, but that it does so with a particularly low sample efficiency (compared to a reference Multi-Layer Perceptron (MLP) model trained with a perceptron). On the other hand the amount of neurons required to achieve a good solution is extremely low compared to the amount of MLP neurons that would be required for solving the problem with a MLP model. This ability of the model to find low-neuron count solutions isn’t of interest for computational reasons (recall the number of parameters per neuron is much higher than in the case of an MLP) but may prove interesting as that low neuron count may make the system easier to interpret.


<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/tcnn_convergence.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Learning curves of policies trained on fully-connected cells with different neuron counts.
</div>
