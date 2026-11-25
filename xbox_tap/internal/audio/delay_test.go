package audio

import "testing"

func TestDelayLineZeroAndDelayed(t *testing.T) {
	d := NewDelayLine()
	in := []int16{10, 20, 30, 40}
	out := make([]int16, 4)

	d.SetDelayMs(0)
	d.Process(in, out)
	for i := range in {
		if out[i] != in[i] {
			t.Fatalf("zero delay mismatch at %d: got %d want %d", i, out[i], in[i])
		}
	}

	// Set 10 ms delay = 480 frames * 2 channels = 960 samples
	d.SetDelayMs(10)
	chunk := make([]int16, 960)
	for i := range chunk {
		chunk[i] = int16(i + 1)
	}
	res1 := make([]int16, 960)
	res2 := make([]int16, 960)
	d.Process(chunk, res1)
	d.Process(make([]int16, 960), res2)
	if res2[0] != 1 || res2[959] != 960 {
		t.Fatalf("10 ms delayed output mismatch: first=%d last=%d", res2[0], res2[959])
	}
}
