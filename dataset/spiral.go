package dataset

import (
	"math"
	"math/rand/v2"

	"gonum.org/v1/gonum/mat"
)

func SpiralData(samples int, classes int) (*mat.Dense, *mat.Dense) {
	X := mat.NewDense(samples*classes, 2, nil)
	y := mat.NewDense(samples*classes, 1, nil)
	X.Zero()
	y.Zero()

	for class := 0; class < classes; class++ {
		for i := samples * class; i < samples*(class+1); i++ {
			r := float64(i) / float64(samples-1) // samples - 1
			t := float64(class*4)*(float64(i)*4)/float64(samples) + rand.NormFloat64()*0.2
			X.Set(i, 0, r*math.Sin(t*2.5))
			X.Set(i, 1, r*math.Cos(t*2.5))
			y.Set(i, 0, float64(class))
		}
	}

	return X, y
}
