# runway

`runway` estimates how long a noisy, oscillating time series takes to reach a
limit: the days until a quantity that grows toward a ceiling fills it. It is a
small Go library with no dependencies beyond the standard library, intended for
humans and agents.

```go
f := runway.New()
for _, p := range points { // p.T time.Time, p.V float64
	if err := f.Observe(p.T, p.V); err != nil {
		// NaN/Inf value, or a time before the last one
	}
}
e, err := f.Forecast(1000) // the ceiling
if errors.Is(err, runway.ErrNoForecast) {
	// the series is flat or heading away; no honest date
} else if err == nil {
	fmt.Printf("%.0f days (%.0f to %.0f), rate %.2f/day, via %s\n",
		e.Days, e.Low, e.High, e.Rate, e.Expert)
}
```

## Why not a single filter

A straight-line fit, or a single Kalman filter, has to pick one reaction speed.
A stiff model is steady but slow: when the real rate changes, it keeps
projecting the old rate for a long time. An agile model turns fast but chases
every wobble, so its date jumps around. On a series that plateaus, accelerates,
or oscillates around a slope, neither setting is right for the whole history.

`runway` keeps several filters at once (the ensemble), each tuned to a different
reaction speed, scores each one on how well it predicted the recent past, and
leans on the recent winner. It turns with the agile expert when the rate
changes and rests on the stiff expert when the rate is steady.

## How it works

Each expert is a two-state Kalman filter tracking a `level` and a `slope` (the
rate in units per day), differing only in the process noise `q` on the slope:
stiff (small `q`) to agile (large `q`). The experts are combined with the
exponentially weighted average forecaster (Hedge): each step, every expert's
one-step prediction error is scored, the scores decay with a half-life for
recency, and the weights are a softmax of the negated decayed scores. Losses are
normalized by a running noise scale, so the weighting does not depend on the
units of the series.

`Forecast(limit)` returns a date only when the series is actually heading for
the limit; a flat or receding series returns `ErrNoForecast`. The `[Low, High]`
band comes from the uncertainty of the rate.

The math and its sources are in the package godoc: Kalman (1960); Welch and
Bishop, "An Introduction to the Kalman Filter"; Bar-Shalom, Li and Kirubarajan
for the process-noise model; Cesa-Bianchi and Lugosi, and Freund and Schapire,
for Hedge.

## Accuracy

Measured on seeded synthetic series with a known full date (`go test -run
TestSynthetic -v`), median absolute error in the predicted days until full, and
the count of dangerous misses (saying the fill is much farther off than it is,
or giving no date when the fill is near):

| series | runway | single stiff Kalman | sliding least squares |
|---|---|---|---|
| rate change (slow then fast) | 0.62 days, 0 dangerous | 17.2 days, 300 dangerous | 3.0 days, 0 dangerous |
| steady with noise | 0.48 days | 0.28 days | - |

Read this honestly. What the ensemble wins is the date error after a rate
change: 0.62 days against 3.0 for least squares and 17.2 for the single stiff
filter. On dangerous misses the ensemble and least squares both score 0 of 300;
the single stiff filter scores 300 of 300, because it keeps projecting the old
rate and says the fill is far off. On a steady series the single stiff filter is
actually a touch better than the ensemble (0.28 against 0.48 days): adaptivity
costs a little when it is not needed. The ensemble earns its place on the series
that change, not on the steady ones.

## Performance

`go test -run xxx -bench . -benchmem` (Apple M4 Pro):

| op | time | allocations |
|---|---|---|
| `Observe` | about 5 ns | 0 |
| `Forecast` | about 8 ns | 0 |

Both are allocation-free on the hot path.

## Options

`New` takes functional options: `WithExperts`, `WithMeasurementNoise`,
`WithConfidence`, `WithLearningRate`, `WithScoreHalfLife`, `WithSelection`
(`Blend` or `Best`), `WithHysteresis`, and `WithMinSamples`. With no options it
uses a three-expert ensemble and documented defaults.

## Known limits

- The confidence band is tight right after a rate change: the filter is briefly
  overconfident until its covariance catches up.
- The predicted days are invariant to the scale of the series, but the band is
  not: it is read off the measurement noise `r`, which is absolute. With the
  default `r` a series whose values are far larger than 1 produces a band that
  collapses toward zero width. Set `WithMeasurementNoise` to the real variance of
  one observation, in the series' own units, for an honest band.
- A store rollback (a value that jumps backward because the source was restored)
  is not modeled; the fallback treats it as a new trend.
- A `Forecaster` holds no global state but is not safe for concurrent use
  without external locking.

## License

Apache 2.0. See `LICENSE` and `NOTICE`.
