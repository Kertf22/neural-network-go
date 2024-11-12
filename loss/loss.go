package loss

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

var STATIC_MINIMUM_VALUE = 1 - 1e-7

// Sparse Labels => INDICES OF THE TARGETS
// One-Hot Labels => MATRIX OF THE TARGETS (1 for the target, 0 for the rest)
func LossClassification(output *mat.Dense, class_targets *mat.Dense) float64 {
	rows, cols := class_targets.Dims()
	values := mat.NewDense(1, rows, nil)
	copyOutput := mat.DenseCopyOf(output)
	copyOutput.Apply(func(i, j int, v float64) float64 {
		if v == 0 {
			return STATIC_MINIMUM_VALUE
		}
		return v
	}, copyOutput)

	for i := 0; i < rows; i++ {
		if cols == 1 {
			c := int(class_targets.At(i, 0))
			values.Set(0, i, -math.Log(copyOutput.At(i, c)))
		} else {
			for j := 0; j < cols; j++ {
				c := class_targets.At(i, j)
				if c != 0 {
					values.Set(0, i, -math.Log(copyOutput.At(i, j)*c))
				}
			}
		}
	}

	return mat.Sum(values) / float64(rows)
}
