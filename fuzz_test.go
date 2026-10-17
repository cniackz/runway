package runway

import (
	"math"
	"testing"
	"time"
)

// hard edge values the fuzzer should mix into v, the limit, and the day steps.
var edgeFloats = []float64{
	0, math.Copysign(0, -1), // +0 and -0
	1, -1, 0.5, -0.5,
	1e-300, 1e300, -1e308, math.MaxFloat64, -math.MaxFloat64,
	math.SmallestNonzeroFloat64, -math.SmallestNonzeroFloat64, // subnormals
	math.NaN(), math.Inf(1), math.Inf(-1),
}

func pick(b byte) float64 { return edgeFloats[int(b)%len(edgeFloats)] }

// checkEstimate enforces the hard invariant: a returned Estimate (err == nil)
// never carries a NaN or Inf, except that High may be +Inf (an open band).
func checkEstimate(t *testing.T, e Estimate, where string) {
	t.Helper()
	for name, x := range map[string]float64{"Days": e.Days, "Low": e.Low, "Rate": e.Rate, "Level": e.Level} {
		if math.IsNaN(x) || math.IsInf(x, 0) {
			t.Fatalf("%s: non-finite %s in estimate without error: %+v", where, name, e)
		}
	}
	if math.IsNaN(e.High) {
		t.Fatalf("%s: NaN High in estimate: %+v", where, e)
	}
}

// FuzzObserveForecast drives a sequence of observations and forecasts with
// arbitrary bytes drawn from the whole float64 class (NaN, infinities, extreme
// finite values, subnormals, signed zeros) in the values, the limit, and the
// day steps. It asserts two things on every step: the forecaster never panics
// and never returns a non-finite Estimate without an error (except High may be
// +Inf), and a bad sample never poisons the state, since a trailing clean ramp
// must forecast finitely.
func FuzzObserveForecast(f *testing.F) {
	f.Add([]byte{1, 2, 3, 4, 5, 6, 7, 8}, byte(2))
	f.Add([]byte{13, 11, 0, 9, 8}, byte(13)) // NaN/Inf values and limit
	f.Add([]byte{9, 9, 9, 9, 9, 9}, byte(9)) // extreme finite values

	f.Fuzz(func(t *testing.T, data []byte, limSel byte) {
		fc := New()
		base := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
		cur := base
		for i := 0; i+1 < len(data); i += 2 {
			// A day step from the edge set (absolute, so repeats and huge gaps
			// both occur), and a value from the edge set.
			stepDays := pick(data[i])
			if math.IsNaN(stepDays) || math.IsInf(stepDays, 0) {
				stepDays = 0
			}
			stepDays = math.Abs(stepDays)
			// Cap the step so cumulative time cannot overflow time.Duration
			// (int64 nanoseconds) over a bounded fuzz input. A large but finite
			// gap is still exercised at this cap.
			if stepDays > 365 {
				stepDays = 365
			}
			cur = cur.Add(time.Duration(stepDays*24) * time.Hour)
			_ = fc.Observe(cur, pick(data[i+1])) // errors fine; a panic is not
			if e, err := fc.Forecast(pick(limSel)); err == nil {
				checkEstimate(t, e, "mid-sequence")
			}
		}

		// Recovery: after any bad input above, a clean daily ramp must leave the
		// forecast finite. A sample may still be declined (if an earlier extreme
		// value grew the state near the float range), but the forecast itself is
		// always finite or a clean sentinel, never a NaN or Inf with err == nil.
		for i := 0; i < 12; i++ {
			cur = cur.Add(24 * time.Hour)
			_ = fc.Observe(cur, 100+10*float64(i))
		}
		switch e, err := fc.Forecast(100000); err {
		case nil:
			checkEstimate(t, e, "after recovery")
		case ErrNoForecast, ErrInsufficientData, ErrInvalidLimit:
			// a clean sentinel is fine
		default:
			t.Fatalf("after recovery: unexpected error %v", err)
		}
	})
}
