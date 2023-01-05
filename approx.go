package approx

import (
	"errors"

	"gonum.org/v1/gonum/integrate/quad"
)

// Cdf returns a wrapper function of gonum/integrate/quad.Fixed.
// The returned function approximates cumulative probabilities of the probability density function.
//
// f is the target pdf. Some pdf parameter settings where pdfs return Inf are NOT allowed (fail to approximate cumulative probabilities).
//
// lb is lower bound of the variables given by the function. -Inf is accepted.
func Cdf(f func(float64) float64, lb float64, n int) func(float64) float64 {
	return func(max float64) float64 { return quad.Fixed(f, lb, max, n, nil, 0) }
}

// Quantile approxiamtes the value of x that satisfies F(X <= x) = p.
//
// cdf is a cumulative distribution function F(x).
//
// p is the probability.
//
// a, b are the lower & upper bounds of the search space (a < b).
//
// eps controls the presicion of the approximation.
// The smaller gives higher presicion and longer computation.
// eps can be 1e-15 at minimum for formal CDF (eps smaller than 1e-15 will causes panic)
// Precision is not ensured depending on types of cdf. Especially, larger eps (such as 1e-10) is recommended for numerical CDF.
func Quantile(cdf func(float64) float64, p, a, b, eps float64) (float64, error) {
	if p <= 0 || 1 <= p {
		panic("Invalid probability. p should be 0 < p < 1.")
	}
	if a > b {
		panic("Invalid bound. `a` should be smaller than b.")
	}
	if eps < 1e-15 {
		panic("Too high precision is required. Please consider larger eps")
	}

	pa := cdf(a)
	pb := cdf(b)
	return quantile(cdf, p, a, b, pa, pb, eps)
}

// Actual implementation of quantile approximation.
// pa, pb are the probability at a and b, respectively.
func quantile(cdf func(float64) float64, p, a, b, pa, pb, eps float64) (float64, error) {
	c := (b + a) / 2
	if b-a < eps {
		return c, nil
	}
	pc := cdf(c)
	if pa <= p && p < pc {
		return quantile(cdf, p, a, c, pa, pc, eps)
	} else if pc <= p && p < pb {
		return quantile(cdf, p, c, b, pc, pb, eps)
	}
	err := errors.New("Value of interest is outside of the range [a, b).")
	return a, err
}
