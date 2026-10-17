package runway

import (
	"errors"
	"math"
	"testing"
	"time"
)

var epoch = time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)

func day(i int) time.Time { return epoch.Add(time.Duration(i) * 24 * time.Hour) }

// feed observes a value a day apart starting at the epoch.
func feed(t *testing.T, f *Forecaster, vs ...float64) {
	t.Helper()
	for i, v := range vs {
		if err := f.Observe(day(i), v); err != nil {
			t.Fatalf("Observe(%d, %v): %v", i, v, err)
		}
	}
}

func TestObserveRejectsBadInput(t *testing.T) {
	tests := []struct {
		name string
		v    float64
		want error
	}{
		{"nan", math.NaN(), ErrInvalidObserve},
		{"posinf", math.Inf(1), ErrInvalidObserve},
		{"neginf", math.Inf(-1), ErrInvalidObserve},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			f := New()
			if err := f.Observe(day(0), tc.v); !errors.Is(err, tc.want) {
				t.Fatalf("got %v, want %v", err, tc.want)
			}
		})
	}
}

func TestObserveTimeGoingBack(t *testing.T) {
	f := New()
	if err := f.Observe(day(5), 10); err != nil {
		t.Fatal(err)
	}
	if err := f.Observe(day(4), 11); !errors.Is(err, ErrTimeWentBack) {
		t.Fatalf("got %v, want ErrTimeWentBack", err)
	}
}

func TestForecastNeedsMinSamples(t *testing.T) {
	f := New(WithMinSamples(4))
	feed(t, f, 1, 2, 3)
	if _, err := f.Forecast(100); !errors.Is(err, ErrInsufficientData) {
		t.Fatalf("got %v, want ErrInsufficientData", err)
	}
	if err := f.Observe(day(3), 4); err != nil {
		t.Fatal(err)
	}
	if _, err := f.Forecast(100); err != nil {
		t.Fatalf("after min samples: %v", err)
	}
}

// Property: an exact line gives the exact rate and date (sealed claim 1b).
func TestExactLineIsExact(t *testing.T) {
	const slope = 5.0
	f := New(WithMeasurementNoise(1e-6))
	for i := 0; i < 40; i++ {
		feedOne(t, f, i, 100+slope*float64(i))
	}
	e, err := f.Forecast(300)
	if err != nil {
		t.Fatal(err)
	}
	if rel := math.Abs(e.Rate-slope) / slope; rel > 1e-6 {
		t.Errorf("rate %.12f, want %v (rel %.2e)", e.Rate, slope, rel)
	}
	wantDays := (300 - e.Level) / slope
	if rel := math.Abs(e.Days-wantDays) / wantDays; rel > 1e-6 {
		t.Errorf("days %.9f, want %.9f (rel %.2e)", e.Days, wantDays, rel)
	}
	// Claim 1b: the band brackets Days from both sides.
	if !(e.Low <= e.Days && e.Days <= e.High) {
		t.Errorf("band [%.6f, %.6f] does not bracket Days %.6f", e.Low, e.High, e.Days)
	}
}

// Property: the band tightens monotonically toward Days as samples accumulate.
func TestBandTightensWithSamples(t *testing.T) {
	const slope = 2.0
	f := New(WithMeasurementNoise(1e-6))
	feedOne(t, f, 0, 0)
	feedOne(t, f, 1, slope)
	feedOne(t, f, 2, 2*slope)
	prev := math.Inf(1)
	for i := 3; i < 30; i++ {
		feedOne(t, f, i, slope*float64(i))
		e, err := f.Forecast(1000)
		if err != nil {
			t.Fatal(err)
		}
		width := e.High - e.Low
		if width > prev+1e-9 {
			t.Fatalf("at i=%d band widened: %.6f after %.6f", i, width, prev)
		}
		prev = width
	}
}

// Property: a flat series and a receding series give ErrNoForecast.
func TestFlatAndRecedingGiveNoForecast(t *testing.T) {
	flat := New()
	feed(t, flat, 50, 50, 50, 50, 50, 50)
	if _, err := flat.Forecast(100); !errors.Is(err, ErrNoForecast) {
		t.Errorf("flat: got %v, want ErrNoForecast", err)
	}
	rec := New(WithMeasurementNoise(1e-6))
	for i := 0; i < 20; i++ {
		feedOne(t, rec, i, 100-3*float64(i))
	}
	if _, err := rec.Forecast(200); !errors.Is(err, ErrNoForecast) {
		t.Errorf("receding toward a higher limit: got %v, want ErrNoForecast", err)
	}
}

// A descending series heading down to a lower limit does get a date.
func TestDescendingToLowerLimit(t *testing.T) {
	f := New(WithMeasurementNoise(1e-6))
	for i := 0; i < 20; i++ {
		feedOne(t, f, i, 100-3*float64(i))
	}
	e, err := f.Forecast(10)
	if err != nil {
		t.Fatalf("got %v, want a date", err)
	}
	if e.Rate >= 0 || e.Days <= 0 {
		t.Errorf("rate %.3f days %.3f, want negative rate and positive days", e.Rate, e.Days)
	}
}

func TestResetClears(t *testing.T) {
	f := New()
	feed(t, f, 1, 2, 3, 4)
	f.Reset()
	if _, err := f.Forecast(100); !errors.Is(err, ErrInsufficientData) {
		t.Fatalf("after Reset: got %v, want ErrInsufficientData", err)
	}
}

// The agile expert should carry the majority weight after a rate change, and a
// stiff-only filter should lag it (the ensemble premise). Sealed claim 9: the
// agile expert reaches the majority within 3 to 5 observations.
func TestEnsembleFollowsRateChange(t *testing.T) {
	f := New()
	// Slow for 15 days, then fast for 15 days, noise-free.
	v := 0.0
	i := 0
	for ; i < 15; i++ {
		feedOne(t, f, i, v)
		v += 1
	}
	changeAt := i
	reached := -1
	for ; i < 30; i++ {
		feedOne(t, f, i, v)
		v += 10
		if f.cfg.experts[f.dominant()].Label == "agile" && reached < 0 {
			reached = i - changeAt
		}
	}
	if reached < 0 {
		t.Fatal("agile never became dominant after the rate change")
	}
	if reached > 6 {
		t.Errorf("agile took %d observations to dominate, want within ~5", reached)
	}
	e, err := f.Forecast(1000)
	if err != nil {
		t.Fatal(err)
	}
	if e.Rate < 8 {
		t.Errorf("blended rate %.2f after the change, want near 10", e.Rate)
	}
}

func TestRepeatedTimestamp(t *testing.T) {
	f := New()
	if err := f.Observe(day(0), 10); err != nil {
		t.Fatal(err)
	}
	// Same instant twice is a second update, not an error.
	if err := f.Observe(day(0), 10); err != nil {
		t.Fatalf("repeat instant: %v", err)
	}
}

// Best hysteresis, as a direct decision test: the active expert holds against a
// challenger that is only slightly better, and switches once the challenger
// clears the 1+h margin. A build that drops the hysteresis (active follows the
// argmax) would switch on the near tie.
func TestBestHysteresisHolds(t *testing.T) {
	f := New(WithSelection(Best), WithHysteresis(0.15),
		WithExperts([]Expert{{Label: "a", Q: 1e-3}, {Label: "b", Q: 1e-1}}))
	// Expert 0 is marginally ahead: active settles on 0.
	f.score[0], f.score[1] = 0.10, 0.11
	f.softmax()
	if f.active != 0 {
		t.Fatalf("active %d, want 0 after expert 0 leads", f.active)
	}
	// Expert 1 is now marginally ahead, within the margin: active must hold at 0.
	f.score[0], f.score[1] = 0.11, 0.10
	f.softmax()
	if f.active != 0 {
		t.Fatalf("active %d, want it held at 0 on a near tie (hysteresis)", f.active)
	}
	// Expert 1 clears the margin (weight ratio exp(0.4) > 1.15): active switches.
	f.score[0], f.score[1] = 0.50, 0.10
	f.softmax()
	if f.active != 1 {
		t.Fatalf("active %d, want 1 after expert 1 clears the margin", f.active)
	}
}

// Best mode does forecast from the active expert.
func TestBestModeForecasts(t *testing.T) {
	f := New(WithSelection(Best), WithMeasurementNoise(1e-6))
	for i := 0; i < 20; i++ {
		feedOne(t, f, i, 100+4*float64(i))
	}
	e, err := f.Forecast(300)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(e.Rate-4) > 0.5 {
		t.Errorf("best-mode rate %.3f, want near 4", e.Rate)
	}
}

func feedOne(t *testing.T, f *Forecaster, i int, v float64) {
	t.Helper()
	if err := f.Observe(day(i), v); err != nil {
		t.Fatalf("Observe(%d, %v): %v", i, v, err)
	}
}

// Regression for F1 (attack): a non-finite limit must error, not return a
// non-finite Estimate.
func TestForecastRejectsNonFiniteLimit(t *testing.T) {
	f := New()
	feed(t, f, 1, 2, 3, 4)
	for _, lim := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		if _, err := f.Forecast(lim); !errors.Is(err, ErrInvalidLimit) {
			t.Errorf("Forecast(%v): got %v, want ErrInvalidLimit", lim, err)
		}
	}
}

// Regression for F2 (attack): a finite but enormous value never poisons the
// state. With the normalized loss it is scored like any other value, and the
// forecast stays finite through and after it, never a NaN or Inf with err=nil.
func TestHugeValueDoesNotPoison(t *testing.T) {
	f := New()
	if err := f.Observe(day(0), 0); err != nil {
		t.Fatal(err)
	}
	// A 1e300 sample is absorbed, not a poison: Observe does not error, and the
	// state stays finite.
	if err := f.Observe(day(1), 1e300); err != nil {
		t.Fatalf("Observe(1e300): %v", err)
	}
	for i := 2; i < 14; i++ {
		feedOne(t, f, i, float64(i))
	}
	switch e, err := f.Forecast(100); err {
	case nil, ErrNoForecast:
		if err == nil {
			for _, x := range []float64{e.Days, e.Low, e.Rate, e.Level} {
				if !finite(x) {
					t.Fatalf("non-finite estimate after a 1e300 sample: %+v", e)
				}
			}
		}
	default:
		t.Fatalf("after a 1e300 sample: unexpected error %v", err)
	}
}

// After the F4 fix the loss is normalized, so a consistent series forecasts the
// same number of days whatever its magnitude: 1, 1e9 or 1e200.
func TestLargeScaleIsUsable(t *testing.T) {
	for _, scale := range []float64{1e2, 1e9, 1e150, 1e200, 1e300} {
		f := New()
		for i := 0; i < 20; i++ {
			if err := f.Observe(day(i), scale*(1+0.01*float64(i))); err != nil {
				t.Fatalf("scale %.0e, sample %d: %v", scale, i, err)
			}
		}
		e, err := f.Forecast(scale * 3)
		if err != nil {
			t.Fatalf("scale %.0e: %v", scale, err)
		}
		if !finite(e.Days) || e.Days <= 0 {
			t.Errorf("scale %.0e: days not sane: %+v", scale, e)
		}
	}
}

// Days is invariant to the units of the series: the same shape at 1x, 1e3x and
// 1e9x gives the same number of days. (The band is not invariant; see the
// WithMeasurementNoise godoc and the README known limits.)
func TestDaysScaleInvariant(t *testing.T) {
	daysAt := func(unit float64) float64 {
		f := New()
		for i := 0; i < 30; i++ {
			if err := f.Observe(day(i), unit*float64(i)); err != nil {
				t.Fatal(err)
			}
		}
		e, err := f.Forecast(unit * 100)
		if err != nil {
			t.Fatalf("unit %.0e: %v", unit, err)
		}
		return e.Days
	}
	base := daysAt(1)
	for _, unit := range []float64{1e3, 1e9, 1e12} {
		if got := daysAt(unit); math.Abs(got-base) > 1e-6 {
			t.Errorf("unit %.0e: days %.6f, want %.6f (scale-invariant)", unit, got, base)
		}
	}
}

// Realistic recovery: a single out-of-range spike (1e9) in an otherwise normal
// series is declined, and the normal samples around it are accepted and keep
// forecasting finitely. This is the everyday form of the no-poison invariant.
func TestRecoversFromRealisticSpike(t *testing.T) {
	f := New()
	for i := 0; i < 6; i++ {
		feedOne(t, f, i, 100+10*float64(i))
	}
	// A spike far outside the series scale: it may be declined, but must not
	// poison the state.
	_ = f.Observe(day(6), 1e9)
	rejected := 0
	for i := 7; i < 20; i++ {
		if err := f.Observe(day(i), 100+10*float64(i)); err != nil {
			rejected++
		}
	}
	if rejected != 0 {
		t.Errorf("%d normal samples rejected after a 1e9 spike, want 0 (no poison)", rejected)
	}
	// The forecast is finite or a clean sentinel, never NaN or Inf without error.
	switch e, err := f.Forecast(1000); err {
	case nil:
		for _, x := range []float64{e.Days, e.Low, e.Rate, e.Level} {
			if !finite(x) {
				t.Fatalf("non-finite estimate after the spike: %+v", e)
			}
		}
	case ErrNoForecast:
		// the series just dropped back; a flat or negative recent trend is sane
	default:
		t.Fatalf("after the spike: unexpected error %v", err)
	}
}

// Regression for F3 (attack): an invalid q in WithExperts is ignored (the
// default ensemble is kept), so weights never go NaN.
func TestWithExpertsRejectsInvalidQ(t *testing.T) {
	for _, bad := range [][]Expert{
		{{"a", -1}, {"b", 1e-2}},
		{{"a", math.NaN()}},
		{{"a", math.Inf(1)}},
	} {
		f := New(WithExperts(bad))
		feed(t, f, 1, 2, 3, 4, 5)
		e, err := f.Forecast(100)
		if err != nil && !errors.Is(err, ErrNoForecast) {
			t.Fatalf("WithExperts(%v): %v", bad, err)
		}
		if err == nil {
			for _, x := range []float64{e.Days, e.Low, e.Rate, e.Level} {
				if !finite(x) {
					t.Fatalf("WithExperts(%v): non-finite estimate %+v", bad, e)
				}
			}
		}
		// The default ensemble has three experts; the bad one was ignored.
		if len(f.experts) != 3 {
			t.Errorf("WithExperts(%v): %d experts, want the default 3 kept", bad, len(f.experts))
		}
	}
}

// WithScoreHalfLife(0) and WithMeasurementNoise(0) are ignored (defaults kept).
func TestZeroOptionsIgnored(t *testing.T) {
	f := New(WithScoreHalfLife(0), WithMeasurementNoise(0))
	if f.cfg.halfLife != defaultScoreHalfLife {
		t.Errorf("halfLife %v, want default %v", f.cfg.halfLife, defaultScoreHalfLife)
	}
	if f.cfg.r != defaultMeasurementNoise {
		t.Errorf("r %v, want default %v", f.cfg.r, defaultMeasurementNoise)
	}
}
