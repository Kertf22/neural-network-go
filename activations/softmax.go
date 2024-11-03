package activations

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

func SoftMax(inputs *mat.Dense) {
	rows, cols := inputs.Dims()
	for i := 0; i < rows; i++ {
		max := mat.Max(inputs.RowView(i))
		for j := 0; j < cols; j++ {
			v := inputs.At(i, j) - max
			inputs.Set(i, j, math.Exp(v))
		}
	}
	for i := 0; i < rows; i++ {
		sum := mat.Sum(inputs.RowView(i))
		for j := 0; j < cols; j++ {
			inputs.Set(i, j, inputs.At(i, j)/sum)
		}
	}
}
