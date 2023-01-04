package approx

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/stat/distuv"
)

func TestCdf(t *testing.T) {
	norm := distuv.Normal{Mu: 0, Sigma: 1}
	gam := distuv.Gamma{Alpha: 1, Beta: 1}
	f := distuv.F{D1: 5, D2: 2}

	data := []struct {
		name string
		f    func(float64) float64
		lb   float64
		max  float64
		n    int
		cdf  func(float64) float64
		tol  float64
	}{
		{"N(0,1)[-Inf,2]", norm.Prob, math.Inf(-1), 2, 300, norm.CDF, 1e-10},
		{"Gam(1,1)[0,2]", gam.Prob, 0, 2, 300, gam.CDF, 1e-10},
		{"Gam(1,1)[0,Inf]", gam.Prob, 0, math.Inf(1), 300, gam.CDF, 1e-10},
		{"F(5,2)[0,10]", f.Prob, 0, 10, 300, f.CDF, 1e-9},
		{"F(5,2)[0,Inf]", f.Prob, 0, math.Inf(1), 300, f.CDF, 1e-9},
	}

	for _, d := range data {
		t.Run(d.name, func(t *testing.T) {
			result := Cdf(d.f, d.lb, d.n)(d.max)
			expected := d.cdf(d.max)
			if math.Abs(result-expected) > d.tol {
				t.Errorf("Expected: %f, Obtained: %f", expected, result)
			}
		})
	}
}

func TestQuantile(t *testing.T) {
	norm := distuv.Normal{Mu: 0, Sigma: 1}
	gam := distuv.Gamma{Alpha: 1, Beta: 1}
	f := distuv.F{D1: 5, D2: 2}
	data := []struct {
		name     string
		cdf      func(float64) float64
		p        float64
		a        float64
		b        float64
		eps      float64
		expected float64
		errMsg   string
	}{
		{"N(0,1),Pr=0.9,eps=1e-2", Cdf(norm.Prob, math.Inf(-1), 300), 0.9, -10, 10, 1e-2, norm.Quantile(0.9), ""},
		{"N(0,1),Pr=0.2,eps=1e-6", Cdf(norm.Prob, math.Inf(-1), 300), 0.2, -10, 10, 1e-6, norm.Quantile(0.2), ""},
		{"N(0,1),Pr=0.3,eps=1e-10", norm.CDF, 0.3, -10, 10, 1e-10, norm.Quantile(0.3), ""},
		{"Gam(1,1),Pr=0.3,eps=1e-6", Cdf(gam.Prob, 0, 300), 0.3, 0, 10, 1e-6, gam.Quantile(0.3), ""},
		{"F(5,2),Pr=0.3,eps=1e-6", Cdf(f.Prob, 0, 300), 0.3, 0, 1, 1e-6, f.Quantile(0.3), ""},
		{"Out_of_range", norm.CDF, 0.9, .001, 1, 1e-6, .001, `Value of interest is outside of the range [a, b].`},
	}

	for _, d := range data {
		t.Run(d.name, func(t *testing.T) {
			result, err := Quantile(d.cdf, d.p, d.a, d.b, d.eps)
			if math.Abs(result-d.expected) > d.eps {
				t.Errorf("Expected: %f, Obtained: %f", d.expected, result)
			}
			var errMsg string
			if err != nil {
				errMsg = err.Error()
			}
			if errMsg != d.errMsg {
				t.Errorf("Expected error message: `%s`, Obtained: `%s`", d.errMsg, errMsg)
			}
		})
	}
}
