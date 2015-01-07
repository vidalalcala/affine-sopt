# affine-sopt

Affine stochastic optimization test. Stochastic optimization using ordinary least squares to approximate the Hessian. The code 
tests a variant of the sequential ordinary least squares estimator developed by Alcala & Goodman with a noise quadratic loss function described
by LeCun, Zhang and Schaul (2013) "No More Pesky Learning Rates." .

Attribution
------------

Please cite [Alcala-Burgos](http://gradworks.umi.com/35/24/3524127.html) if you find
this code useful in your research and add your paper to the 
[testimonials list](http://github.com/vidalalcala/affine-sopt/testimonial.md)

Requirements
------------

* [Torch7 library](http://torch.ch/) 

Running
-------

	$ th affine-sopt-quadratic
