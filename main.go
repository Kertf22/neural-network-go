package main

import (
	// "fmt"
	"fmt"
	"image/color"
	"math/rand"

	"neural-network-go/activation"
	"neural-network-go/acuracy"
	"neural-network-go/dataset"
	"neural-network-go/layer"
	"neural-network-go/loss"
	"time"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

var colors = []color.RGBA{
	{R: 255, A: 255},         // red
	{G: 255, A: 255},         // green
	{B: 255, A: 255},         // blue
	{R: 255, G: 255, A: 255}, // yellow
	{R: 255, B: 255, A: 255}, // magenta
}

func denseToPoints(dense *mat.Dense, classes int, class int) plotter.XYs {
	offset := dense.RawMatrix().Rows / classes
	points := make(plotter.XYs, offset)

	for i := 0; i < offset; i++ {
		points[i].X = dense.At(class*offset+i, 0)
		points[i].Y = dense.At(class*offset+i, 1)
	}
	return points
}

func MakePlot(x *mat.Dense, classes int) {
	p := plot.New()
	p.Title.Text = "Points"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"
	p.Add(plotter.NewGrid())
	for i := 0; i < classes; i++ {
		s, err := plotter.NewScatter(denseToPoints(x, classes, i))
		if err != nil {
			panic(err)
		}
		s.Color = colors[i]
		p.Add(s)
	}
	if err := p.Save(5*vg.Inch, 5*vg.Inch, "chards/"+
		time.Now().Local().Format(time.RFC3339)+"pointss.png"); err != nil {
		panic(err)
	}
}

func RunSpiralModel() {
	classes := 3
	x, y := dataset.SpiralData(100, classes)
	// MakePlot(x, classes)
	//
	l := layer.Layer{}
	l.InitLayer(2, 3)

	output := l.Foward(x)
	activation.ReLu(output)

	l2 := layer.Layer{}
	l2.InitLayer(3, 3)

	output = l2.Foward(output)
	activation.SoftMax(output)

	lossValue := loss.LossClassification(output, y)
	acc := acuracy.Acuracy(output, y)
	fmt.Println("loss", lossValue)
	fmt.Println("acuracy", acc)
}

func RunVerticalModel() {
	classes := 3
	x, y := dataset.VerticalData(100, classes)
	l1 := layer.Layer{}
	l1.InitLayer(2, 3)
	l2 := layer.Layer{}
	l2.InitLayer(3, 3)
	lossValue, acc := RunModel(x, y, &l1, &l2)

	fmt.Println("loss", lossValue)
	fmt.Println("acuracy", acc)
}

func RunModel(x *mat.Dense, y *mat.Dense, l1 *layer.Layer, l2 *layer.Layer) (float64, float64) {
	output := l1.Foward(x)
	active := activation.ReLu(output)
	output = l2.Foward(active)
	active2 := activation.SoftMax(output)

	lossValue := loss.LossClassification(active2, y)
	acc := acuracy.Acuracy(active2, y)
	return lossValue, acc
}

func Optimizer() {
	classes := 3
	x, y := dataset.VerticalData(100, classes)
	l1 := layer.Layer{}
	l1.InitLayer(2, 3)
	l2 := layer.Layer{}
	l2.InitLayer(3, 3)

	lowestLoss, _ := RunModel(x, y, &l1, &l2)

	bestLayer1 := l1.CopyOf()
	bestLayer2 := l2.CopyOf()

	for i := 0; i < 10000; i++ {
		UpdateRandomDense(l1.Weights)
		UpdateRandomDense(l1.Bias)
		UpdateRandomDense(l2.Weights)
		UpdateRandomDense(l2.Bias)

		lossValue, acc := RunModel(x, y, &l1, &l2)

		if lowestLoss > lossValue {
			bestLayer1 = l1.CopyOf()
			bestLayer2 = l2.CopyOf()
			lowestLoss = lossValue
			fmt.Println("New best model found at iteration", i, "with loss:", lossValue, "and accuracy:", acc)
		} else {
			l1 = *bestLayer1.CopyOf()
			l2 = *bestLayer2.CopyOf()
		}
	}
}

func UpdateRandomDense(dense *mat.Dense) {
	rows, cols := dense.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			v := dense.At(i, j) + 0.05*rand.Float64()
			dense.Set(i, j, v)
		}
	}
}
func f(x float64) float64 {
	return 2 * x * x
}

func derivateFromInput(dvalues, weights *mat.Dense) *mat.Dense {
	var r mat.Dense
	r.Mul(dvalues, weights)

	return &r
}

func derivateFromWeights(dvalues, inputs *mat.Dense) *mat.Dense {
	var r mat.Dense
	r.Mul(dvalues.T(), inputs)

	return &r
}

func derivateFromBias(dvalues *mat.Dense) *mat.Dense {
	var r mat.Dense = *mat.NewDense(1, dvalues.RawMatrix().Cols, nil)

	for i := 0; i < dvalues.RawMatrix().Rows; i++ {
		rowSum := mat.Sum(dvalues.ColView(i))
		r.Set(0, i, rowSum)
	}
	return &r
}

func main() {
	// dvalues := mat.NewDense(3, 3, []float64{
	// 	1, 1, 1,
	// 	2, 2, 2,
	// 	3, 3, 3,
	// })

	inputs := mat.DenseCopyOf(mat.NewDense(3, 4, []float64{
		1, 2, 3, 2.5,
		2.0, 5.0, -1.0, 2.0,
		-1.5, 2.7, 3.3, -0.8,
	}))

	layer1 := layer.Layer{
		Weights: mat.DenseCopyOf(mat.NewDense(3, 4, []float64{
			0.2, 0.8, -0.5, 1,
			0.5, -0.91, 0.26, -0.5,
			-0.26, -0.27, 0.17, 0.87,
		}).T()),
		Bias: mat.NewDense(1, 3, []float64{2, 3, 0.5}),
	}

	output := layer1.Foward(inputs)
	
	relu_output := activation.ReLu(output)

	drelu := activation.ReluBackward(output, relu_output)
	
	_, dweights, dbias := layer1.Backward(drelu, inputs)

	fmt.Println(mat.Formatted(dweights))

	dweights.Scale(0.001, dweights)
	dbias.Scale(0.001, dbias)

	layer1.Weights.Sub(layer1.Weights, dweights.T())
	layer1.Bias.Sub(layer1.Bias, dbias)
	fmt.Println(mat.Formatted(layer1.Weights))
	fmt.Println(mat.Formatted(layer1.Bias))

}
