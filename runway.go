// Package runway estimates how long a noisy, oscillating time series takes to
// reach a limit, for example the days until a quantity that grows toward a
// ceiling fills it.
//
// It runs an ensemble of local linear-trend Kalman filters, each tuned to a
// different amount of process noise, and each step it leans on whichever expert
// has predicted the recent window best. A single fixed filter must choose
// between being steady but slow to react to a real change of rate and being
// fast but chasing noise; the ensemble keeps several and follows the recent
// winner.
//
// # Model
//
// Each expert is a two-state Kalman filter with state [level, slope], where
// slope is the rate in units per day. Over a step of dt days the state
// propagates with F = [[1, dt], [0, 1]] and the continuous white-noise
// acceleration process noise
//
//	Q(dt) = q * [[dt^3/3, dt^2/2], [dt^2/2, dt]]
//
// (Bar-Shalom, Li and Kirubarajan, Estimation with Applications to Tracking and
// Navigation, 2001, section 6.2.2). The measurement is of the level,
// H = [1, 0], with variance r. Predict and update are the standard Kalman
// equations (R. E. Kalman, 1960, A New Approach to Linear Filtering and
// Prediction Problems; G. Welch and G. Bishop, An Introduction to the Kalman
// Filter, UNC TR 95-041).
//
// # Ensemble
//
// Experts are combined with the exponentially weighted average forecaster, also
// known as Hedge (N. Cesa-Bianchi and G. Lugosi, Prediction, Learning, and
// Games, 2006, chapter 2; Y. Freund and R. Schapire, 1997, A Decision-Theoretic
// Generalization of On-Line Learning and an Application to Boosting). Each step
// the one-step prediction error of every expert is scored, the scores decay
// with a half-life for recency, and the weights are a softmax of the negated
// decayed scores. Losses are normalized by a running noise scale so the
// weighting does not depend on the units of the series.
//
// A Forecaster holds no global state and is not safe for concurrent use without
// external locking; there is no hidden mutex on the hot path.
package runway

import (
	"errors"
	"math"
	"time"
)

// Errors returned by a Forecaster. All are sentinels, comparable with
// errors.Is.
var (
	// ErrInsufficientData means Forecast was called before the minimum number
	// of observations set by WithMinSamples.
	ErrInsufficientData = errors.New("runway: not enough observations yet")
	// ErrNoForecast means the series is not heading for the limit: the rate is
	// flat, or it points away from the limit, so there is no honest date.
	ErrNoForecast = errors.New("runway: series is not heading for the limit")
	// ErrInvalidObserve means the observed value cannot be folded in: it is NaN
	// or Inf, or it is so large that its squared prediction error overflows and
	// would leave the state non-finite. The sample is rejected and the state is
	// left unchanged.
	ErrInvalidObserve = errors.New("runway: value is not usable (NaN, Inf, or too large to score)")
	// ErrTimeWentBack means an observation time is before the last one.
	ErrTimeWentBack = errors.New("runway: observation time is before the last one")
	// ErrInvalidLimit means the limit passed to Forecast is NaN or Inf.
	ErrInvalidLimit = errors.New("runway: limit is NaN or Inf")
)

// Selection is how the ensemble turns per-expert states into one forecast.
type Selection int

const (
	// Blend averages every expert by weight. It is smooth because it averages.
	Blend Selection = iota
	// Best uses only the highest-weight expert, with hysteresis so the active
	// expert does not flap between two near-tied experts.
	Best
)

// Expert is one member of the ensemble: a label for reporting and the process
// noise q on its slope, per day. A small q is stiff (a nearly constant rate,
// smooth, slow to turn); a large q is agile (the rate can move each step, fast
// to turn, noisier).
type Expert struct {
	Label string
	Q     float64
}

// Estimate is the result of a Forecast.
type Estimate struct {
	Days   float64 // days from the last observation until the limit is reached
	Low    float64 // optimistic bound, fewer days, from the rate band
	High   float64 // pessimistic bound, more days; +Inf if the slow side is open
	Rate   float64 // units per day used for the estimate
	Level  float64 // filtered level at the last observation
	Expert string  // label of the dominant expert
}

// Defaults used by New when an option is not set.
const (
	defaultMeasurementNoise = 1.0
	defaultConfidence       = 1.0
	defaultLearningRate     = 1.0
	defaultScoreHalfLife    = 7.0
	defaultMinSamples       = 3
	defaultHysteresis       = 0.15

	initLevelVar = 1.0 // P[0][0] seed: the first sample pins the level to ~r
	initSlopeVar = 1e3 // P[1][1] seed: the slope is unknown, so start wide
	scaleFloor   = 1e-12
)

// defaultExperts is the ensemble New uses when WithExperts is not given: three
// experts from stiff to agile. The rationale and tuning are in the README.
func defaultExperts() []Expert {
	return []Expert{
		{Label: "stiff", Q: 1e-6},
		{Label: "mid", Q: 1e-3},
		{Label: "agile", Q: 1e-1},
	}
}

type config struct {
	experts    []Expert
	r          float64
	z          float64
	eta        float64
	halfLife   float64
	sel        Selection
	hyst       float64
	minSamples int
}

// Option configures a Forecaster. Pass options to New.
type Option func(*config)

// finitePos reports whether x is finite and strictly positive. Options use it
// so a NaN or an infinity never slips through a bare x > 0 test (Inf > 0 is
// true).
func finitePos(x float64) bool { return x > 0 && !math.IsInf(x, 1) }

// WithExperts sets the ensemble. Each entry is one expert with its process
// noise q, which must be finite and not negative. An empty slice, or any entry
// with an invalid q, is ignored and the default ensemble is kept.
func WithExperts(e []Expert) Option {
	return func(c *config) {
		if len(e) == 0 {
			return
		}
		for _, x := range e {
			if x.Q < 0 || math.IsNaN(x.Q) || math.IsInf(x.Q, 0) {
				return // reject the whole ensemble; keep the default
			}
		}
		c.experts = append([]Expert(nil), e...)
	}
}

// WithMeasurementNoise sets r, the variance of a single observation, in the
// squared units of the series. The default is 1. A value that is not finite and
// positive is ignored.
//
// The predicted number of days is invariant to the scale of the series, but the
// [Low, High] band is not: it is read off r, which is absolute. With the default
// r a series whose values are far larger than 1 (bytes, say, rather than
// gigabytes) produces a band that collapses toward zero width, which reads as
// overconfident. For an honest band on such a series, set r to the real variance
// of one observation in the series' own units.
func WithMeasurementNoise(r float64) Option {
	return func(c *config) {
		if finitePos(r) {
			c.r = r
		}
	}
}

// WithConfidence sets z, the number of standard deviations for the Low and High
// band. The default is 1 (one sigma). A value that is not finite and positive
// is ignored.
func WithConfidence(z float64) Option {
	return func(c *config) {
		if finitePos(z) {
			c.z = z
		}
	}
}

// WithLearningRate overrides eta, the Hedge learning rate on the normalized
// losses. The default is 1. It is meant for experiments; the default is tuned
// for the normalized loss scale. A value that is not finite and positive is
// ignored.
func WithLearningRate(eta float64) Option {
	return func(c *config) {
		if finitePos(eta) {
			c.eta = eta
		}
	}
}

// WithScoreHalfLife sets the half-life in days of the loss decay that gives the
// weighting its recency. The default is 7. A value that is not finite and
// positive is ignored.
func WithScoreHalfLife(days float64) Option {
	return func(c *config) {
		if finitePos(days) {
			c.halfLife = days
		}
	}
}

// WithSelection sets Blend (the default) or Best.
func WithSelection(m Selection) Option {
	return func(c *config) { c.sel = m }
}

// WithHysteresis sets the Best switch margin h: the active expert changes only
// when a challenger's weight exceeds the incumbent's by the factor 1+h. The
// default is 0.15. A negative value is ignored.
func WithHysteresis(h float64) Option {
	return func(c *config) {
		if h >= 0 {
			c.hyst = h
		}
	}
}

// WithMinSamples sets how many observations must arrive before Forecast returns
// a value instead of ErrInsufficientData. The default is 3. A value below 2 is
// raised to 2, since a slope needs two points.
func WithMinSamples(n int) Option {
	return func(c *config) {
		if n < 2 {
			n = 2
		}
		c.minSamples = n
	}
}

// expertState is the runtime Kalman state of one expert. P is symmetric, so
// p01 == p10 and only three entries are kept.
type expertState struct {
	q             float64
	level, slope  float64
	p00, p01, p11 float64
}

func (e *expertState) reset(q float64) {
	e.q = q
	e.level, e.slope = 0, 0
	e.p00, e.p01, e.p11 = initLevelVar, 0, initSlopeVar
}

// predicted returns the state after advancing dt days, without mutating e. The
// predicted level is the one-step prediction scored against the next sample.
func (e *expertState) predicted(dt float64) (level, p00, p01, p11 float64) {
	// x' = F x; the slope is unchanged by a constant-velocity predict.
	level = e.level + dt*e.slope
	// P' = F P F^T + Q(dt), with F = [[1,dt],[0,1]] and p10 == p01.
	m00 := e.p00 + 2*dt*e.p01 + dt*dt*e.p11
	m01 := e.p01 + dt*e.p11
	p00 = m00 + e.q*dt*dt*dt/3
	p01 = m01 + e.q*dt*dt/2
	p11 = e.p11 + e.q*dt
	return level, p00, p01, p11
}

// Forecaster is an ensemble of Kalman experts over one series. Create it with
// New. It is not safe for concurrent use.
type Forecaster struct {
	cfg     config
	experts []expertState
	score   []float64 // decayed normalized loss per expert
	weight  []float64 // current normalized weights
	scale   float64   // running noise scale s2
	active  int       // dominant expert index, for Best hysteresis
	last    time.Time
	n       int
	seeded  bool // scale has been seeded

	// Scratch, preallocated so Observe is allocation-free for any ensemble
	// size. Observe computes the next state here, validates it is finite, and
	// commits only then, so a bad sample cannot poison the live state.
	sq     []float64
	nscore []float64
	nlevel []float64
	nslope []float64
	np00   []float64
	np01   []float64
	np11   []float64
}

// New returns a Forecaster ready to use. With no options it uses the default
// three-expert ensemble and the documented default constants. The returned
// value is usable immediately; there is no global state.
func New(opts ...Option) *Forecaster {
	cfg := config{
		experts:    defaultExperts(),
		r:          defaultMeasurementNoise,
		z:          defaultConfidence,
		eta:        defaultLearningRate,
		halfLife:   defaultScoreHalfLife,
		sel:        Blend,
		hyst:       defaultHysteresis,
		minSamples: defaultMinSamples,
	}
	for _, o := range opts {
		o(&cfg)
	}
	f := &Forecaster{cfg: cfg}
	n := len(cfg.experts)
	f.experts = make([]expertState, n)
	f.score = make([]float64, n)
	f.weight = make([]float64, n)
	f.sq = make([]float64, n)
	f.nscore = make([]float64, n)
	f.nlevel = make([]float64, n)
	f.nslope = make([]float64, n)
	f.np00 = make([]float64, n)
	f.np01 = make([]float64, n)
	f.np11 = make([]float64, n)
	f.Reset()
	return f
}

// Reset clears all observations and restores the initial uniform weights.
func (f *Forecaster) Reset() {
	for i := range f.experts {
		f.experts[i].reset(f.cfg.experts[i].Q)
		f.score[i] = 0
		f.weight[i] = 1 / float64(len(f.experts))
	}
	f.scale = 0
	f.active = 0
	f.n = 0
	f.seeded = false
	f.last = time.Time{}
}

// Observe folds one sample into every expert. The time t must not be before the
// last observed time; a repeat of the same instant applies a second measurement
// update without advancing time.
//
// Observe never leaves the state non-finite: it computes the next state, checks
// that every part of it is finite, and commits only then. A NaN or Inf value,
// or a value so large that its squared prediction error overflows, is rejected
// with ErrInvalidObserve and the state is left unchanged, so one bad sample
// cannot poison later forecasts.
func (f *Forecaster) Observe(t time.Time, v float64) error {
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return ErrInvalidObserve
	}
	if f.n == 0 {
		for i := range f.experts {
			f.experts[i].level = v
		}
		f.last = t
		f.n = 1
		return nil
	}
	if t.Before(f.last) {
		return ErrTimeWentBack
	}
	dt := t.Sub(f.last).Hours() / 24
	rho := math.Pow(0.5, dt/f.cfg.halfLife)

	// Predict and score into scratch, and compute the post-update state into
	// scratch, without touching the live state yet.
	//
	// The loss only ranks experts and is normalized by a running scale, so it
	// must not overflow regardless of the magnitude of the series. The scale is
	// an exponentially decayed mean of the absolute innovation, and the loss is
	// the squared normalized innovation (e/scale)^2. Normalizing before squaring,
	// with a denominator at least this step's mean absolute innovation, bounds
	// each normalized innovation by the expert count, so no finite value can
	// overflow the loss or poison the weights.
	n := float64(len(f.experts))
	var sumAbs float64
	for i := range f.experts {
		pl, pp00, pp01, pp11 := f.experts[i].predicted(dt)
		e := v - pl
		f.sq[i] = e // raw innovation; normalized to a loss below
		sumAbs += math.Abs(e)
		// Measurement update of the level from the predicted state.
		s := pp00 + f.cfg.r
		k0 := pp00 / s
		k1 := pp01 / s
		y := v - pl
		f.nlevel[i] = pl + k0*y
		f.nslope[i] = f.experts[i].slope + k1*y
		f.np00[i] = (1 - k0) * pp00
		f.np01[i] = (1 - k0) * pp01
		f.np11[i] = pp11 - k1*pp01
	}
	meanAbs := sumAbs / n
	newScale := meanAbs
	if f.seeded {
		newScale = rho*f.scale + (1-rho)*meanAbs
	}
	// The denominator never drops below this step's scale, so a sudden jump that
	// dwarfs the established scale still normalizes to a bounded loss.
	denom := math.Max(newScale, meanAbs) + scaleFloor
	for i := range f.experts {
		u := f.sq[i] / denom
		f.nscore[i] = rho*f.score[i] + u*u
	}

	// Validate: reject the sample if anything is not finite, state untouched.
	if !finite(meanAbs) || !finite(newScale) ||
		!allFinite(f.nscore) || !allFinite(f.nlevel) || !allFinite(f.nslope) ||
		!allFinite(f.np00) || !allFinite(f.np01) || !allFinite(f.np11) {
		return ErrInvalidObserve
	}

	// Commit.
	for i := range f.experts {
		f.experts[i].level = f.nlevel[i]
		f.experts[i].slope = f.nslope[i]
		f.experts[i].p00 = f.np00[i]
		f.experts[i].p01 = f.np01[i]
		f.experts[i].p11 = f.np11[i]
		f.score[i] = f.nscore[i]
	}
	f.scale = newScale
	f.seeded = true
	f.softmax()
	f.last = t
	f.n++
	return nil
}

func finite(x float64) bool { return !math.IsNaN(x) && !math.IsInf(x, 0) }

func allFinite(a []float64) bool {
	for _, x := range a {
		if !finite(x) {
			return false
		}
	}
	return true
}

// softmax recomputes weights = softmax(-eta * score) with max-subtraction for
// stability, and tracks the dominant expert with Best hysteresis.
func (f *Forecaster) softmax() {
	best := 0
	for i := 1; i < len(f.score); i++ {
		if f.score[i] < f.score[best] {
			best = i
		}
	}
	// -eta*score, shifted so the max is 0 (the min score is the max of -score).
	shift := -f.cfg.eta * f.score[best]
	var sum float64
	for i := range f.score {
		w := math.Exp(-f.cfg.eta*f.score[i] - shift)
		f.weight[i] = w
		sum += w
	}
	for i := range f.weight {
		f.weight[i] /= sum
	}
	// Best hysteresis: switch the active expert only when a challenger beats the
	// incumbent's weight by the factor 1+h.
	if f.weight[best] > f.weight[f.active]*(1+f.cfg.hyst) {
		f.active = best
	}
}

// Forecast estimates when the series reaches lim from the latest state. It
// returns ErrInvalidLimit if lim is not finite, ErrInsufficientData before
// WithMinSamples observations, and ErrNoForecast when the rate does not lead to
// the limit within a finite number of days. A returned Estimate is always
// finite except that High may be +Inf when the slow side of the band does not
// head for the limit.
func (f *Forecaster) Forecast(lim float64) (Estimate, error) {
	if math.IsNaN(lim) || math.IsInf(lim, 0) {
		return Estimate{}, ErrInvalidLimit
	}
	if f.n < f.cfg.minSamples {
		return Estimate{}, ErrInsufficientData
	}
	level, rate, slopeVar, label := f.combine()

	gap := lim - level
	// Not heading for the limit: a flat rate, or a rate pointing away.
	if gap > 0 && rate <= 0 || gap < 0 && rate >= 0 || rate == 0 {
		return Estimate{}, ErrNoForecast
	}
	if gap == 0 {
		return Estimate{Days: 0, Low: 0, High: 0, Rate: rate, Level: level, Expert: label}, nil
	}

	days := gap / rate
	// A rate so small the date overflows is no honest date.
	if !finite(days) {
		return Estimate{}, ErrNoForecast
	}
	sd := math.Sqrt(math.Max(slopeVar, 0))
	// The band evaluates days at the rate plus or minus z sigmas, clamped to the
	// side that still heads for the limit. The faster rate gives fewer days.
	fast := rate + math.Copysign(f.cfg.z*sd, rate)
	slow := rate - math.Copysign(f.cfg.z*sd, rate)
	low := gap / fast
	high := math.Inf(1)
	// slow still heads for the limit only if it keeps rate's sign.
	if (rate > 0 && slow > 0) || (rate < 0 && slow < 0) {
		high = gap / slow
	}
	if low < 0 || !finite(low) {
		low = 0
	}
	return Estimate{Days: days, Low: low, High: high, Rate: rate, Level: level, Expert: label}, nil
}

// combine returns the level, rate, slope variance, and dominant label according
// to the selection mode.
func (f *Forecaster) combine() (level, rate, slopeVar float64, label string) {
	dom := f.dominant()
	label = f.cfg.experts[dom].Label
	if f.cfg.sel == Best {
		e := &f.experts[dom]
		return e.level, e.slope, e.p11, label
	}
	// Blend: weighted mean of level and slope, and the mixture variance of the
	// slope (within-expert variance plus the spread of the slopes).
	for i := range f.experts {
		level += f.weight[i] * f.experts[i].level
		rate += f.weight[i] * f.experts[i].slope
	}
	for i := range f.experts {
		d := f.experts[i].slope - rate
		slopeVar += f.weight[i] * (f.experts[i].p11 + d*d)
	}
	return level, rate, slopeVar, label
}

// dominant is the expert whose state the report names: the active one under
// Best, the highest weight under Blend.
func (f *Forecaster) dominant() int {
	if f.cfg.sel == Best {
		return f.active
	}
	best := 0
	for i := 1; i < len(f.weight); i++ {
		if f.weight[i] > f.weight[best] {
			best = i
		}
	}
	return best
}
