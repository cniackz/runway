package runway

import (
	"testing"
	"time"
)

func BenchmarkObserve(b *testing.B) {
	f := New()
	base := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	// Warm up so the hot path is steady (scale seeded, weights settled).
	for i := 0; i < 8; i++ {
		_ = f.Observe(base.Add(time.Duration(i)*24*time.Hour), float64(i))
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		t := base.Add(time.Duration(8+i) * 24 * time.Hour)
		_ = f.Observe(t, float64(8+i))
	}
}

func BenchmarkFilterPredict(b *testing.B) {
	f, err := NewFilter(0, 1e-3, 1)
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = f.Predict(1)
		_ = f.Update(float64(i)) // keep the covariance from growing without bound
	}
}

func BenchmarkFilterUpdate(b *testing.B) {
	f, err := NewFilter(0, 1e-3, 1)
	if err != nil {
		b.Fatal(err)
	}
	_ = f.Predict(1)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = f.Update(float64(i % 100))
	}
}

func BenchmarkForecast(b *testing.B) {
	f := New()
	base := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	for i := 0; i < 20; i++ {
		_ = f.Observe(base.Add(time.Duration(i)*24*time.Hour), float64(i)*3)
	}
	b.ReportAllocs()
	b.ResetTimer()
	var sink float64
	for i := 0; i < b.N; i++ {
		e, _ := f.Forecast(1000)
		sink += e.Days
	}
	_ = sink
}
