import scipy.optimize as so


def _optimize(im, target_class, epsilon, maxiter, verbose):

    image = im
    min_, max_ = (0, 255)

    # store the shape for later and operate on the flattened image
    shape = image.shape
    dtype = image.dtype
    image = image.flatten().astype(np.float64)

    n = len(image)
    bounds = [(min_, max_)] * n

    x0 = image

    def lbfgsb(c):
	approx_grad_eps = (max_ - min_) / 100
	x, f, d = so.fmin_l_bfgs_b(
	    loss,
	    x0,
	    args=(c,),
	    approx_grad=self._approximate_gradient,
	    bounds=bounds,
	    m=15,
	    maxiter=maxiter,
	    epsilon=approx_grad_eps)


    c = epsilon
    for i in range(30):
	c = 2 * c
	is_adversarial = lbfgsb(c)
	if verbose:
	    print 'Tested c = {:.4e}: {}'.format(c,
		('adversarial' if is_adversarial else 'not adversarial'))
	if is_adversarial:
	    break

    else:  # pragma: no cover

        if verbose:
            print 'Could not find an adversarial; maybe the model returns wrong gradients'  # noqa: E501
        return

    # binary search
    c_low = 0
    c_high = c
    while c_high - c_low >= epsilon:
        c_half = (c_low + c_high) / 2
        is_adversarial = lbfgsb(c_half)
        if verbose:
            print 'Tested c = {:.4e}: {} ({:.4e}, {:.4e})'.format(
                c_half,
                ('adversarial' if is_adversarial else 'not adversarial'),
                c_low,
                c_high)
        if is_adversarial:
            c_high = c_half
        else:
            c_low = c_half

