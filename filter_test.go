package runway

import (
	"errors"
	"math"
	"testing"
)

const covTol = 1e-9

// psd reports whether P is numerically positive semidefinite, with a small
// relative tolerance on the determinant for floating-point drift.
func psd(p00, p01, p11 float64) bool {
	if p00 < 0 || p11 < 0 {
		return false
	}
	det := p00*p11 - p01*p01
	tol := covTol * (1 + p00*p11)
	return det >= -tol
}

// Claim 3: a Filter stepped with Predict then Update reproduces, bit for bit,
// the state a one-expert ensemble reaches with Observe. This pins the ensemble
// onto the exported Filter.
func TestFilterReproducesEnsembleStep(t *testing.T) {
	const q = 1e-3
	f := New(WithExperts([]Expert{{Label: "x", Q: q}}))
	feedOne(t, f, 0, 100) // seeds level = 100
	feedOne(t, f, 1, 103) // one predict+update over dt = 1 day
	ex := f.experts[0]

	// The ensemble's default measurement noise is 1.
	fl, err := NewFilter(100, q, defaultMeasurementNoise)
	if err != nil {
		t.Fatal(err)
	}
	if err := fl.Predict(1); err != nil {
		t.Fatal(err)
	}
	if err := fl.Update(103); err != nil {
		t.Fatal(err)
	}
	p00, p01, p11 := fl.Cov()
	if fl.Level() != ex.level || fl.Slope() != ex.slope ||
		p00 != ex.p00 || p01 != ex.p01 || p11 != ex.p11 {
		t.Fatalf("filter %v/%v/%v,%v,%v != ensemble %v/%v/%v,%v,%v",
			fl.Level(), fl.Slope(), p00, p01, p11,
			ex.level, ex.slope, ex.p00, ex.p01, ex.p11)
	}
}

// Claim 4: P stays symmetric (one stored p01) and positive semidefinite after
// every Predict and every Update.
func TestFilterCovStaysPSD(t *testing.T) {
	f, err := NewFilter(0, 1e-2, 2)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 200; i++ {
		if err := f.Predict(0.5); err != nil {
			t.Fatal(err)
		}
		if p00, p01, p11 := f.Cov(); !psd(p00, p01, p11) {
			t.Fatalf("after predict %d: not PSD: %v %v %v", i, p00, p01, p11)
		}
		z := math.Sin(float64(i)) * 10 // an oscillating, unbiased signal
		if err := f.Update(z); err != nil {
			t.Fatal(err)
		}
		if p00, p01, p11 := f.Cov(); !psd(p00, p01, p11) {
			t.Fatalf("after update %d: not PSD: %v %v %v", i, p00, p01, p11)
		}
	}
}

// Claim 5: Predict(0) is an exact no-op on level, slope and covariance.
func TestPredictZeroIsExactNoOp(t *testing.T) {
	f, err := NewFilter(42, 1e-1, 1, WithInitialCovariance(3, 1, 5))
	if err != nil {
		t.Fatal(err)
	}
	_ = f.Update(40) // move it off the seed
	before := *f.Clone()
	if err := f.Predict(0); err != nil {
		t.Fatal(err)
	}
	p00, p01, p11 := f.Cov()
	if f.Level() != before.level || f.Slope() != before.slope ||
		p00 != before.p00 || p01 != before.p01 || p11 != before.p11 {
		t.Fatalf("Predict(0) changed the state: %+v -> %v/%v/%v,%v,%v",
			before, f.Level(), f.Slope(), p00, p01, p11)
	}
}

// Claim 6: independent oracle. On a noiseless line z = a + b*t sampled daily,
// the filter converges to (a, b) and predicts the line ahead.
func TestFilterTracksExactLine(t *testing.T) {
	const a, b = 100.0, 3.0
	f, err := NewFilter(a, 1e-2, 1)
	if err != nil {
		t.Fatal(err)
	}
	last := 0.0
	for i := 1; i <= 30; i++ {
		if err := f.Predict(1); err != nil {
			t.Fatal(err)
		}
		last = a + b*float64(i)
		if err := f.Update(last); err != nil {
			t.Fatal(err)
		}
	}
	if math.Abs(f.Slope()-b) > 1e-6 {
		t.Errorf("slope %.9f, want %.9f", f.Slope(), b)
	}
	if math.Abs(f.Level()-last) > 1e-6 {
		t.Errorf("level %.9f, want %.9f", f.Level(), last)
	}
	// Predicting h days ahead follows the line.
	h := 10.0
	g := f.Clone()
	if err := g.Predict(h); err != nil {
		t.Fatal(err)
	}
	want := last + b*h
	// A tiny residual slope error is amplified by the horizon h, so the forward
	// tolerance is h times the slope tolerance, not the 1e-6 of the fitted point.
	if math.Abs(g.Level()-want) > 1e-4 {
		t.Errorf("predicted level %.9f, want %.9f", g.Level(), want)
	}
}

// Claim 8: NewFilter rejects every invalid parameter with ErrInvalidFilter and
// returns no filter.
func TestNewFilterValidation(t *testing.T) {
	bad := []struct {
		name        string
		level, q, r float64
		opts        []FilterOption
	}{
		{"negative q", 0, -1, 1, nil},
		{"NaN q", 0, math.NaN(), 1, nil},
		{"Inf q", 0, math.Inf(1), 1, nil},
		{"zero r", 0, 1e-3, 0, nil},
		{"negative r", 0, 1e-3, -1, nil},
		{"NaN r", 0, 1e-3, math.NaN(), nil},
		{"Inf r", 0, 1e-3, math.Inf(1), nil},
		{"NaN level", math.NaN(), 1e-3, 1, nil},
		{"Inf level", math.Inf(-1), 1e-3, 1, nil},
		{"cov negative p00", 0, 1e-3, 1, []FilterOption{WithInitialCovariance(-1, 0, 1)}},
		{"cov negative p11", 0, 1e-3, 1, []FilterOption{WithInitialCovariance(1, 0, -1)}},
		{"cov negative det", 0, 1e-3, 1, []FilterOption{WithInitialCovariance(1, 2, 1)}},
		{"cov NaN", 0, 1e-3, 1, []FilterOption{WithInitialCovariance(math.NaN(), 0, 1)}},
		{"cov Inf", 0, 1e-3, 1, []FilterOption{WithInitialCovariance(1, 0, math.Inf(1))}},
	}
	for _, c := range bad {
		f, err := NewFilter(c.level, c.q, c.r, c.opts...)
		if !errors.Is(err, ErrInvalidFilter) {
			t.Errorf("%s: err = %v, want ErrInvalidFilter", c.name, err)
		}
		if f != nil {
			t.Errorf("%s: got a filter, want nil", c.name)
		}
	}
	// A valid PSD covariance is accepted.
	if _, err := NewFilter(0, 1e-3, 1, WithInitialCovariance(2, 1, 3)); err != nil {
		t.Errorf("valid covariance rejected: %v", err)
	}
}

// WithInitialSlope seeds the slope, defaults to 0, and rejects a non-finite
// value.
func TestWithInitialSlope(t *testing.T) {
	// Default is 0.
	f, err := NewFilter(0, 1e-3, 1)
	if err != nil {
		t.Fatal(err)
	}
	if f.Slope() != 0 {
		t.Errorf("default slope %v, want 0", f.Slope())
	}
	// A finite value is stored exactly.
	for _, s := range []float64{3, -2.5, 1e6, math.SmallestNonzeroFloat64} {
		g, err := NewFilter(0, 1e-3, 1, WithInitialSlope(s))
		if err != nil {
			t.Fatalf("WithInitialSlope(%v): %v", s, err)
		}
		if g.Slope() != s {
			t.Errorf("slope %v, want %v", g.Slope(), s)
		}
	}
	// A non-finite value is rejected and returns no filter.
	for _, s := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		g, err := NewFilter(0, 1e-3, 1, WithInitialSlope(s))
		if !errors.Is(err, ErrInvalidFilter) || g != nil {
			t.Errorf("WithInitialSlope(%v): err %v, filter %v", s, err, g)
		}
	}
	// A seeded slope is used: predicting ahead moves by slope*dt.
	h, err := NewFilter(100, 1e-9, 1, WithInitialSlope(5))
	if err != nil {
		t.Fatal(err)
	}
	if err := h.Predict(10); err != nil {
		t.Fatal(err)
	}
	if math.Abs(h.Level()-150) > 1e-9 {
		t.Errorf("level after predict %v, want 150", h.Level())
	}
}

// Predict and Update reject invalid input and leave the state unchanged.
func TestFilterStepValidation(t *testing.T) {
	f, err := NewFilter(10, 1e-3, 1)
	if err != nil {
		t.Fatal(err)
	}
	before := *f.Clone()
	if err := f.Predict(-1); !errors.Is(err, ErrTimeWentBack) {
		t.Errorf("Predict(-1): %v, want ErrTimeWentBack", err)
	}
	if err := f.Predict(math.NaN()); !errors.Is(err, ErrInvalidObserve) {
		t.Errorf("Predict(NaN): %v, want ErrInvalidObserve", err)
	}
	if err := f.Update(math.Inf(1)); !errors.Is(err, ErrInvalidObserve) {
		t.Errorf("Update(Inf): %v, want ErrInvalidObserve", err)
	}
	if p00, p01, p11 := f.Cov(); f.Level() != before.level || f.Slope() != before.slope ||
		p00 != before.p00 || p01 != before.p01 || p11 != before.p11 {
		t.Fatalf("a rejected step changed the state")
	}
}
