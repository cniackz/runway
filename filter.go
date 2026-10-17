package runway

import (
	"errors"
	"math"
)

// ErrInvalidFilter is returned by NewFilter when a parameter is out of range: a
// negative, NaN or infinite q; a non-positive, NaN or infinite r; a non-finite
// initial level; or an initial covariance that is not finite and positive
// semidefinite.
var ErrInvalidFilter = errors.New("runway: invalid filter parameter")

// Filter is a two-state Kalman filter tracking a level and a slope (the rate in
// units per day) under a constant-velocity model with white-acceleration process
// noise of intensity q per day. It is the per-expert model the ensemble is built
// from, exported so a caller can assemble its own selector on top of it.
//
// The zero value is not usable; create a Filter with NewFilter. A Filter is not
// safe for concurrent use.
type Filter struct {
	q, r          float64
	level, slope  float64
	p00, p01, p11 float64
}

// FilterOption configures a Filter at construction. An option that rejects its
// input returns an error, which NewFilter passes back.
type FilterOption func(*Filter) error

// NewFilter returns a filter at the given initial level with slope 0. q is the
// process-noise intensity (finite and at least 0); r is the variance of one
// measurement (finite and greater than 0). The initial covariance defaults to
// the same seed the ensemble uses; override it with WithInitialCovariance. Any
// invalid parameter returns ErrInvalidFilter and a nil filter, never a poisoned
// one.
func NewFilter(level, q, r float64, opts ...FilterOption) (*Filter, error) {
	if !finite(level) || !(finite(q) && q >= 0) || !finitePos(r) {
		return nil, ErrInvalidFilter
	}
	f := &Filter{q: q, r: r, level: level, p00: initLevelVar, p11: initSlopeVar}
	for _, o := range opts {
		if err := o(f); err != nil {
			return nil, err
		}
	}
	return f, nil
}

// WithInitialSlope sets the initial slope, in units per day; the default is 0.
// A caller that already has a rate (from the first two samples, say) can seed it
// here. A non-finite value returns ErrInvalidFilter.
func WithInitialSlope(s float64) FilterOption {
	return func(f *Filter) error {
		if !finite(s) {
			return ErrInvalidFilter
		}
		f.slope = s
		return nil
	}
}

// WithInitialCovariance sets the initial covariance P. It must be symmetric
// (p10 is taken equal to p01) and positive semidefinite: p00 >= 0, p11 >= 0 and
// the determinant p00*p11 - p01^2 >= 0, all finite. Otherwise NewFilter returns
// ErrInvalidFilter.
func WithInitialCovariance(p00, p01, p11 float64) FilterOption {
	return func(f *Filter) error {
		if !finite(p00) || !finite(p01) || !finite(p11) ||
			p00 < 0 || p11 < 0 || p00*p11-p01*p01 < 0 {
			return ErrInvalidFilter
		}
		f.p00, f.p01, f.p11 = p00, p01, p11
		return nil
	}
}

// Predict advances the filter dt days under the constant-velocity model, with
// dt finite and not negative. It returns ErrTimeWentBack for a negative dt and
// ErrInvalidObserve for a non-finite dt or a non-finite result; in every error
// case the filter is left unchanged.
func (f *Filter) Predict(dt float64) error {
	if math.IsNaN(dt) || math.IsInf(dt, 0) {
		return ErrInvalidObserve
	}
	if dt < 0 {
		return ErrTimeWentBack
	}
	nl, np00, np01, np11 := predictLevelCov(f.level, f.slope, f.p00, f.p01, f.p11, f.q, dt)
	if !finite(nl) || !finite(np00) || !finite(np01) || !finite(np11) {
		return ErrInvalidObserve
	}
	f.level, f.p00, f.p01, f.p11 = nl, np00, np01, np11
	return nil
}

// Update folds one measurement z into the filter with the standard Kalman
// update (H = [1, 0], variance r). It returns ErrInvalidObserve for a non-finite
// z or a non-finite result, and leaves the filter unchanged in that case.
func (f *Filter) Update(z float64) error {
	if math.IsNaN(z) || math.IsInf(z, 0) {
		return ErrInvalidObserve
	}
	nl, ns, np00, np01, np11 := updateLevelCov(f.level, f.slope, f.p00, f.p01, f.p11, f.r, z)
	if !finite(nl) || !finite(ns) || !finite(np00) || !finite(np01) || !finite(np11) {
		return ErrInvalidObserve
	}
	f.level, f.slope, f.p00, f.p01, f.p11 = nl, ns, np00, np01, np11
	return nil
}

// Level is the current filtered level.
func (f *Filter) Level() float64 { return f.level }

// Slope is the current filtered slope, in units per day.
func (f *Filter) Slope() float64 { return f.slope }

// Cov returns the current covariance entries; p10 equals p01.
func (f *Filter) Cov() (p00, p01, p11 float64) { return f.p00, f.p01, f.p11 }

// Clone returns an independent copy of the filter, useful for predicting ahead
// without disturbing the live state.
func (f *Filter) Clone() *Filter { c := *f; return &c }

// predictLevelCov is the constant-velocity time update: x' = F x and
// P' = F P F^T + Q(dt), with F = [[1, dt], [0, 1]], p10 == p01, and the
// white-acceleration process noise Q(dt) = q [[dt^3/3, dt^2/2], [dt^2/2, dt]].
// The slope is unchanged by the predict. This is the single source of truth for
// the time update, shared by Filter.Predict and the ensemble.
func predictLevelCov(level, slope, p00, p01, p11, q, dt float64) (nl, np00, np01, np11 float64) {
	nl = level + dt*slope
	m00 := p00 + 2*dt*p01 + dt*dt*p11
	m01 := p01 + dt*p11
	np00 = m00 + q*dt*dt*dt/3
	np01 = m01 + q*dt*dt/2
	np11 = p11 + q*dt
	return nl, np00, np01, np11
}

// updateLevelCov is the standard scalar measurement update with H = [1, 0] and
// variance r, applied to a predicted state (predLevel, slope, pp..). It is the
// single source of truth for the measurement update, shared by Filter.Update and
// the ensemble.
func updateLevelCov(predLevel, slope, pp00, pp01, pp11, r, z float64) (nl, ns, np00, np01, np11 float64) {
	s := pp00 + r
	k0 := pp00 / s
	k1 := pp01 / s
	y := z - predLevel
	nl = predLevel + k0*y
	ns = slope + k1*y
	np00 = (1 - k0) * pp00
	np01 = (1 - k0) * pp01
	np11 = pp11 - k1*pp01
	return nl, ns, np00, np01, np11
}
