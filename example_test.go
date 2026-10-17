package runway_test

import (
	"errors"
	"fmt"
	"time"

	"github.com/cniackz/runway"
)

// The common case: feed daily samples of a growing quantity and ask how many
// days until it reaches a limit.
func ExampleForecaster_Forecast() {
	f := runway.New()
	start := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	// A quantity climbing by about 10 a day toward a ceiling of 1000.
	used := []float64{400, 409, 421, 430, 438, 451, 460, 472, 480, 491}
	for i, v := range used {
		if err := f.Observe(start.Add(time.Duration(i)*24*time.Hour), v); err != nil {
			panic(err)
		}
	}
	e, err := f.Forecast(1000)
	if err != nil {
		panic(err)
	}
	fmt.Printf("about %.0f days, between %.0f and %.0f\n", e.Days, e.Low, e.High)
	// Output:
	// about 50 days, between 49 and 52
}

// A flat series has no honest date.
func ExampleForecaster_Forecast_flat() {
	f := runway.New()
	start := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	for i := 0; i < 8; i++ {
		_ = f.Observe(start.Add(time.Duration(i)*24*time.Hour), 500)
	}
	_, err := f.Forecast(1000)
	fmt.Println(errors.Is(err, runway.ErrNoForecast))
	// Output:
	// true
}
