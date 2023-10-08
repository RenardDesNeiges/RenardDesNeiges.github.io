---
layout: page
title: Trajectory optimization for AWE
description: Trajectory optimization for a fixed-wing airborne wind energy system.
img: assets/img/traj_opt.gif
importance: 3
category: work
---

Airborne wind energy (AWE) is an emerging field in renewable power which intends to harvest wind energy using airborne devices. They do so by exploiting the relative velocity between an airmass (the wind) and the ground to create a traction forces that can in turn be converted into electrical power.

<div class="row center">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/AWE.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    A schematic representation of the AWE system we study.
</div>

There are two main AWE power transmission concepts: lift mode and drag mode. In drag mode, an airborne turbine is used to generate electrical power in the air, that electrical power is then transmitted to the ground through the tether. In lift mode a kite is flying in crosswind patterns while pulling on a tether which drives a generator on the ground. Lift mode AWE concepts are further separated into rigid kite concepts, where the "kite" has a design which is close to a con- ventional airplane and has onboard control surfaces, and soft kite concepts where the kite is closer to a hobby kite. In this project we study a rigid kite, lift mode AWE system, more specifically we work with a model of the prototype small scale AWE system. We aim to develop robust methods for the generation of power-optimal (in the sense that flying them yields a maximum average generated power) trajectories for a rigid wing, glider-like, lift mode AWE system.

We have a system of differential algebraic equations that provide a dynamical model of the AWE system. These allow us to compute system trajectories given a sequence of control inputs and an initial point.

<div class="row center">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/traj_opt.gif" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    An example trajectories computed using the model (here without a thether).
</div>

In turn we formulate an optimization problem in which we are minimizing some loss (the negative energy production) while satisfying a set of constraints (periodicity constraints, control input constraints, and that the system moves according to its dynamics).

Because of the long time-horizon (≈ 15s) required to solve the periodic optimization problem for time-optimal trajectories, shooting methods are unpractical. We thus use a direct multi-segment orthogonal collocation method to discretize the problem.
In practice most of our results were achieved with 2nd degree polynomial, and multiple segments, which is also sometimes referred to as [Hermite-Simpson collocation](https://epubs.siam.org/doi/10.1137/16M1062569).

The resulting transcription is thereupon solved using sequential quadratic programming using the non-linear program solver [IPOPT](https://github.com/coin-or/Ipopt), and the sparse linear solver [MUMPS](https://mumps-solver.org/index.php) for the underlying linear programs.


The problem obtained from the transcription yields a non-linear program whose solutions correspond to a physically meaningful result. However, in practice, because of the strong non-linearity of the problem most initialization of the solver lead to an infeasible local minimum, this is particularly true when including the tether model described in section and when using the full model rather than the kinematic approximation.

To overcome this limitation we choose to implement a homotopy procedure. The general idea behind that approach is to pick a less strongly non-linear optimization problem close to the one we want to solve and use the solution of that easier to solve problem as an initial guess for the full model. This is performed recursively as we gradually add in the strongly non-linear dynamics to a simplified model until we have a feasible solution for the strongly non-linear model. Essentially we start with "open" feedback loops and then close them gradually.    

<div class="row center">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/homotopy_pos.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Sequence of trajectories (position over time), generated as the homotopy algorithm progresses. 
</div>

Using the homotopy procedure we have an algorithm able to generate power-optimal trajectories for AWE.  

<div class="row center">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/flypath_power_traj_scaled.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Power optimal trajectory generated by the algorithm. 
</div>