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
// eps controls the presicion of the estimation.
// The smaller gives higher presicion and longer computation.
func Quantile(cdf func(float64) float64, p, a, b, eps float64) (float64, error) {
	if p <= 0 || 1 <= p {
		panic("Invalid probability. p should be 0 < p < 1.")
	}
	if a > b {
		panic("Invalid bound. a should be smaller than b.")
	}

	c := (b + a) / 2
	if b-a < eps {
		return c, nil
	}
	pa := cdf(a)
	pb := cdf(b)
	pc := cdf(c)
	if pa <= p && p < pc {
		return Quantile(cdf, p, a, c, eps)
	} else if pc <= p && p < pb {
		return Quantile(cdf, p, c, b, eps)
	}
	err := errors.New("Value of interest is outside of the range [a, b].")
	return a, err
}
