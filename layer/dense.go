package layer

import (
	"math/rand/v2"

	"gonum.org/v1/gonum/mat"
)

type Layer struct {
	Weights *mat.Dense
	Bias    *mat.Dense
}

// inputs per sample
// neurons of the layer
func (layer *Layer) InitLayer(inputs int, neurons int) {
	// layer.Weights = mat.NewDense(inputs, neurons, nil) //
	layer.Weights = mat.NewDense(inputs, neurons, nil) // TRANSPOSED
	layer.Bias = mat.NewDense(1, neurons, nil)

	for i := 0; i < inputs; i++ {
		for j := 0; j < neurons; j++ {
			layer.Weights.Set(i, j, 0.01*(rand.Float64()-0.5)*2)
		}
	}

}

func (layer *Layer) Foward(inputs *mat.Dense) *mat.Dense {

	// rows - number of samples
	// columns - number os neurons of a layer
	rows, _ := inputs.Dims()
	_, cols := layer.Weights.Dims()
	result := mat.NewDense(rows, cols, nil)
	result.Product(inputs, layer.Weights)

	for i := 0; i < result.RawMatrix().Rows; i++ {
		for j := 0; j < result.RawMatrix().Cols; j++ {
			result.Set(i, j,
				result.At(i, j)+layer.Bias.At(0, j))
		}
	}

	return result
}

func FlatMatrix(m [][]float64) []float64 {
	var flat []float64
	for i := 0; i < len(m); i++ {
		flat = append(flat, m[i]...)
	}
	return flat
}

func FromDense(dense *mat.Dense) [][]float64 {
	r := make([][]float64, dense.RawMatrix().Rows)
	for i := 0; i < dense.RawMatrix().Rows; i++ {
		line := make([]float64, dense.RawMatrix().Cols)
		for j := 0; j < dense.RawMatrix().Cols; j++ {
			line[j] = dense.At(i, j)
		}
		r[i] = line
	}
	return r
}
