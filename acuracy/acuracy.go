package acuracy

import "gonum.org/v1/gonum/mat"

// how often the largest confidence is the correct class
// in terms of a fraction
func Acuracy(output *mat.Dense, class_targets *mat.Dense) float64 {
	rows, cols := class_targets.Dims()
	max := 0
	for i := 0; i < rows; i++ {
		c := 0
		if cols == 1 {
			c = int(mat.Max(class_targets.RowView(i)))
		} else {
			for j := 0; j < cols; j++ {
				if class_targets.At(i, j) == 1 {
					c = j
					break
				}
			}
		}
		if mat.Max(output.RowView(i)) == output.At(i, c) {
			max++
		}
	}

	return float64(max) / float64(rows)
}
