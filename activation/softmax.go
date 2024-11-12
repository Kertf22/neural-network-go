package activation

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

// type SoftMax struct {
// 	inputs *mat.Dense
// 	output *mat.Dense
// }

// func (softmax *SoftMax) Foward()

func SoftMax(inputs *mat.Dense) *mat.Dense {
	copy := mat.DenseCopyOf(inputs)
	rows, cols := copy.Dims()
	for i := 0; i < rows; i++ {
		max := mat.Max(copy.RowView(i))
		for j := 0; j < cols; j++ {
			v := copy.At(i, j) - max
			copy.Set(i, j, math.Exp(v))
		}
	}
	for i := 0; i < rows; i++ {
		sum := mat.Sum(copy.RowView(i))
		for j := 0; j < cols; j++ {
			copy.Set(i, j, copy.At(i, j)/sum)
		}
	}
	return copy
}
