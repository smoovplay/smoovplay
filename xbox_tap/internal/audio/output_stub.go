//go:build !windows

package audio

type noopSink struct{}

// OpenDefaultSink returns a no-op audio sink on non-Windows platforms.
func OpenDefaultSink(sampleRate, channels int) (Sink, error) {
	return noopSink{}, nil
}

func (noopSink) WritePCM([]byte) error { return nil }
func (noopSink) Close() error          { return nil }
