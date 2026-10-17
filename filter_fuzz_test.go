package runway

import (
	"math"
	"testing"
)

// FuzzFilter drives NewFilter, Predict and Update with parameters drawn from the
// whole float64 class (NaN, infinities, extreme finite values, subnormals,
// signed zeros). It asserts three things: no call ever panics; a Predict or
// Update that returns nil never leaves the state non-finite; and a call that
// returns an error never changes the state, compared against a Clone taken
// before it.
func FuzzFilter(f *testing.F) {
	f.Add(byte(2), byte(2), byte(2), []byte{2, 3, 4, 5})
	f.Add(byte(13), byte(9), byte(7), []byte{13, 14, 15, 0}) // NaN q, extreme steps
	f.Add(byte(0), byte(4), byte(8), []byte{9, 9, 9, 9})     // signed zeros, big values

	f.Fuzz(func(t *testing.T, li, qi, ri byte, data []byte) {
		fl, err := NewFilter(pick(li), pick(qi), pick(ri))
		if err != nil {
			if fl != nil {
				t.Fatalf("error returned a non-nil filter")
			}
			return // invalid construction is a valid outcome
		}
		// A constructed filter always starts finite.
		assertFinite(t, fl, "after NewFilter")

		for i := 0; i+1 < len(data); i += 2 {
			before := *fl.Clone()
			if err := fl.Predict(pick(data[i])); err != nil {
				assertUnchanged(t, fl, before, "rejected Predict")
			} else {
				assertFinite(t, fl, "after Predict")
			}
			before = *fl.Clone()
			if err := fl.Update(pick(data[i+1])); err != nil {
				assertUnchanged(t, fl, before, "rejected Update")
			} else {
				assertFinite(t, fl, "after Update")
			}
		}
	})
}

func assertFinite(t *testing.T, f *Filter, where string) {
	t.Helper()
	p00, p01, p11 := f.Cov()
	for _, x := range []float64{f.Level(), f.Slope(), p00, p01, p11} {
		if math.IsNaN(x) || math.IsInf(x, 0) {
			t.Fatalf("%s: non-finite state %v/%v/%v,%v,%v", where, f.Level(), f.Slope(), p00, p01, p11)
		}
	}
}

func assertUnchanged(t *testing.T, f *Filter, before Filter, where string) {
	t.Helper()
	p00, p01, p11 := f.Cov()
	if f.Level() != before.level || f.Slope() != before.slope ||
		p00 != before.p00 || p01 != before.p01 || p11 != before.p11 {
		t.Fatalf("%s mutated the state", where)
	}
}
