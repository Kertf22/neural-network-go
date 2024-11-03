package activations

import "gonum.org/v1/gonum/mat"

func ReLu(inputs *mat.Dense) {
	for i := 0; i < inputs.RawMatrix().Rows; i++ {
		for j := 0; j < inputs.RawMatrix().Cols; j++ {
			v := inputs.At(i, j)
			if v < 0 {
				inputs.Set(i, j, 0)
			}
		}
	}
}
