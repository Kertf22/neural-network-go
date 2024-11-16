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

func (layer *Layer) CopyOf() *Layer {
	return &Layer{
		Weights: mat.DenseCopyOf(layer.Weights),
		Bias:    mat.DenseCopyOf(layer.Bias),
	}
}

// gradient is the derivate of next layer with respect to inputs
func (layer *Layer) Backward(gradient, inputs *mat.Dense) (*mat.Dense, *mat.Dense, *mat.Dense) {
	// derivate of weights is inputs transposed
	var dweights mat.Dense
	dweights.Mul(gradient.T(), inputs)
	// derivate of bias is 1, so we can just sum all the values
	var dbias mat.Dense
	dbias.Mul(mat.NewDense(1, gradient.RawMatrix().Cols, nil), gradient)
	// derivate of inputs is weights transposed
	var lgradient mat.Dense
	lgradient.Mul(gradient, layer.Weights.T())

	return &lgradient, &dweights, &dbias
}
