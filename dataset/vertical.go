package dataset

import (
	"math/rand/v2"

	"gonum.org/v1/gonum/mat"
)

// import numpy as np

// # Modified from:
// # Copyright (c) 2015 Andrej Karpathy
// # License: https://github.com/cs231n/cs231n.github.io/blob/master/LICENSE
// # Source: https://cs231n.github.io/neural-networks-case-study/
// def create_data(samples, classes):
//     X = np.zeros((samples*classes, 2))
//     y = np.zeros(samples*classes, dtype='uint8')
//     for class_number in range(classes):
//         ix = range(samples*class_number, samples*(class_number+1))
//         X[ix] = np.c_[np.random.randn(samples)*.1 + (class_number)/3, np.random.randn(samples)*.1 + 0.5]
//         y[ix] = class_number
//     return X, y

func VerticalData(samples int, classes int) (*mat.Dense, *mat.Dense) {
	X := mat.NewDense(samples*classes, 2, nil)
	y := mat.NewDense(samples*classes, 1, nil)
	X.Zero()
	y.Zero()

	for class := 0; class < classes; class++ {
		for i := samples * class; i < samples*(class+1); i++ {
			X.Set(i, 0, rand.NormFloat64()*0.1+float64(class)/3)
			X.Set(i, 1, rand.NormFloat64()*0.1+0.5)
			y.Set(i, 0, float64(class))
		}
	}

	return X, y
}
