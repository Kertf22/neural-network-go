package activation

import "gonum.org/v1/gonum/mat"

func ReLu(inputs *mat.Dense) *mat.Dense {
	copy := mat.DenseCopyOf(inputs)
	r, c := copy.Dims()

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			v := copy.At(i, j)
			if v < 0 {
				copy.Set(i, j, 0)
			}
		}
	}
	return copy
}

func ReluBackward(z, dvalues *mat.Dense) *mat.Dense {
	var r mat.Dense = *mat.NewDense(dvalues.RawMatrix().Rows, dvalues.RawMatrix().Cols, nil)

	for i := 0; i < dvalues.RawMatrix().Rows; i++ {
		for j := 0; j < dvalues.RawMatrix().Cols; j++ {
			if z.At(i, j) > 0 {
				r.Set(i, j, dvalues.At(i, j))
			}
		}
	}
	return &r
}
