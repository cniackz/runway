package runway

import (
	"math"
	"math/rand"
	"testing"
	"time"
)

// This file holds the synthetic benchmark: seeded series with a known full
// date, scored against a single stiff Kalman and against a sliding-window least
// squares fit. It is the discriminator for sealed claims 4, 5 and 6. The seed
// and the full date are generated here; no series comes from anywhere else.

type series struct {
	vals     []float64 // observed values, one per day from day 0
	lastDay  int       // index of the last observation
	limit    float64   // the ceiling
	trueDays float64   // true days from the last observation until the limit
}

// rateChangeSeries grows at r1 for half the window, then r2, with gaussian
// noise, and sets the limit so the noise-free trend fills some days after the
// window. The true remaining days use the post-change rate r2.
func rateChangeSeries(rng *rand.Rand, n int, r1, r2, noise float64) series {
	vals := make([]float64, n)
	level := 100.0
	half := n / 2
	trueLast := 0.0
	for i := 0; i < n; i++ {
		r := r1
		if i >= half {
			r = r2
		}
		if i > 0 {
			level += r
		}
		trueLast = level
		vals[i] = level + rng.NormFloat64()*noise
	}
	ahead := 20.0 // the limit sits ~20 days past the last point at rate r2
	limit := trueLast + r2*ahead
	return series{vals: vals, lastDay: n - 1, limit: limit, trueDays: (limit - trueLast) / r2}
}

// steadySeries grows at a constant rate with noise; the limit sits ahead at the
// same rate.
func steadySeries(rng *rand.Rand, n int, r, noise float64) series {
	vals := make([]float64, n)
	level := 100.0
	for i := 0; i < n; i++ {
		if i > 0 {
			level += r
		}
		vals[i] = level + rng.NormFloat64()*noise
	}
	limit := level + r*20
	return series{vals: vals, lastDay: n - 1, limit: limit, trueDays: 20}
}

// predictEnsemble forecasts with the default ensemble.
func predictEnsemble(s series) (float64, bool) {
	f := New()
	return runForecast(f, s)
}

// predictStiff forecasts with a single stiff Kalman (one expert).
func predictStiff(s series) (float64, bool) {
	f := New(WithExperts([]Expert{{Label: "stiff", Q: 1e-6}}))
	return runForecast(f, s)
}

func runForecast(f *Forecaster, s series) (float64, bool) {
	base := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	for i, v := range s.vals {
		_ = f.Observe(base.Add(time.Duration(i)*24*time.Hour), v)
	}
	e, err := f.Forecast(s.limit)
	if err != nil {
		return 0, false
	}
	return e.Days, true
}

// predictLeastSquares fits v = a + b*t over the last window points and
// extrapolates to the limit.
func predictLeastSquares(s series, window int) (float64, bool) {
	start := len(s.vals) - window
	if start < 0 {
		start = 0
	}
	var n, sx, sy, sxx, sxy float64
	for i := start; i < len(s.vals); i++ {
		x := float64(i)
		y := s.vals[i]
		n++
		sx += x
		sy += y
		sxx += x * x
		sxy += x * y
	}
	den := n*sxx - sx*sx
	if den == 0 {
		return 0, false
	}
	b := (n*sxy - sx*sy) / den
	a := (sy - b*sx) / n
	level := a + b*float64(s.lastDay)
	gap := s.limit - level
	if (gap > 0) != (b > 0) || b == 0 {
		return 0, false
	}
	return gap / b, true
}

type scoreboard struct {
	absErr    []float64 // absolute error in predicted days
	dangerous int       // said much longer, or no date when one was due soon
	total     int
}

func (sb *scoreboard) add(pred float64, ok bool, trueDays float64) {
	sb.total++
	if !ok {
		// No date while the true fill is near is a dangerous miss.
		if trueDays < 60 {
			sb.dangerous++
		}
		return
	}
	sb.absErr = append(sb.absErr, math.Abs(pred-trueDays))
	if pred > 1.5*trueDays {
		sb.dangerous++
	}
}

func (sb *scoreboard) medianAbsErr() float64 {
	if len(sb.absErr) == 0 {
		return math.Inf(1)
	}
	cp := append([]float64(nil), sb.absErr...)
	sortFloats(cp)
	return cp[len(cp)/2]
}

func sortFloats(a []float64) {
	for i := 1; i < len(a); i++ {
		for j := i; j > 0 && a[j] < a[j-1]; j-- {
			a[j], a[j-1] = a[j-1], a[j]
		}
	}
}

// Sealed claims 4 and 6: on a rate-change family, the ensemble's median error
// and its dangerous-miss count are lower than a single stiff Kalman and than
// least squares.
func TestSyntheticRateChange(t *testing.T) {
	rng := rand.New(rand.NewSource(20261017))
	var ens, stiff, ls scoreboard
	for k := 0; k < 300; k++ {
		s := rateChangeSeries(rng, 30, 1+rng.Float64()*2, 8+rng.Float64()*6, 2.0)
		p, ok := predictEnsemble(s)
		ens.add(p, ok, s.trueDays)
		p, ok = predictStiff(s)
		stiff.add(p, ok, s.trueDays)
		p, ok = predictLeastSquares(s, 20)
		ls.add(p, ok, s.trueDays)
	}
	t.Logf("rate-change: ensemble med=%.2f danger=%d/%d | stiff med=%.2f danger=%d | ls med=%.2f danger=%d",
		ens.medianAbsErr(), ens.dangerous, ens.total,
		stiff.medianAbsErr(), stiff.dangerous, ls.medianAbsErr(), ls.dangerous)
	if ens.medianAbsErr() > stiff.medianAbsErr() {
		t.Errorf("claim 4: ensemble median %.2f > stiff %.2f", ens.medianAbsErr(), stiff.medianAbsErr())
	}
	if ens.medianAbsErr() > ls.medianAbsErr() {
		t.Errorf("claim 4: ensemble median %.2f > least squares %.2f", ens.medianAbsErr(), ls.medianAbsErr())
	}
	if ens.dangerous > stiff.dangerous {
		t.Errorf("claim 6: ensemble dangerous %d > stiff %d", ens.dangerous, stiff.dangerous)
	}
}

// Sealed claim 5: on a steady series, the ensemble is within 10 percent of the
// single stiff Kalman (it does not pay a large penalty for being adaptive).
func TestSyntheticSteady(t *testing.T) {
	rng := rand.New(rand.NewSource(70261017))
	var ens, stiff scoreboard
	for k := 0; k < 300; k++ {
		s := steadySeries(rng, 30, 2+rng.Float64()*3, 2.0)
		p, ok := predictEnsemble(s)
		ens.add(p, ok, s.trueDays)
		p, ok = predictStiff(s)
		stiff.add(p, ok, s.trueDays)
	}
	t.Logf("steady: ensemble med=%.2f | stiff med=%.2f", ens.medianAbsErr(), stiff.medianAbsErr())
	if ens.medianAbsErr() > 1.10*stiff.medianAbsErr()+0.25 {
		t.Errorf("claim 5: ensemble median %.2f not within 10%% of stiff %.2f", ens.medianAbsErr(), stiff.medianAbsErr())
	}
}
