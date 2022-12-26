# Loss Surface Simplexes 


This repository contains the code for Bayesian Machine Learning course at NYU CSCI-GA.3033-​087. We explored the loss landscape on top of paper [Loss Surface Simplexes for Mode Connecting Volumes and Fast Ensembling](https://arxiv.org/abs/2102.13042) (ICML 2021) by Greg Benton, Wesley Maddox, Sanae Lotfi, and Andrew Gordon Wilson. 

The contemporary view of loss landscape structure is that individual SGD solutions exist on a connected multi-dimensional low loss volume in weight space. While previous works have shown the existence of this volume using modes obtained via vanilla SGD training, we extend it by including sharp modes and sharp mode-connecting points by using poison training.  Our experiments suggest that sharp minima (a) lie around the boundary of well-generalizing basins, and are not isolated but rather connected to each other both via a (b) low-loss surface which generalizes well, and (c) low-loss surface which generalizes poorly. 
