package main

import (
	"fmt"
	"image/color"
	"math"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

type Layer struct {
	weights [][]float64
	bias    []float64
}

func MatrixProduct(ma *mat.Dense, mb *mat.Dense) *mat.Dense {
	ad, _ := ma.Dims()
	_, bd := mb.T().Dims()
	result := mat.NewDense(ad, bd, nil)
	result.Mul(ma, mb.T())
	return result
}

func flatMatrix(m [][]float64) []float64 {
	var flat []float64
	for i := 0; i < len(m); i++ {
		flat = append(flat, m[i]...)
	}
	return flat
}

func runLayer(inputs [][]float64, layer Layer) *mat.Dense {
	flatInputs := flatMatrix(inputs)
	flatWeights := flatMatrix(layer.weights)

	dimsX := len(layer.bias)
	dimsY := len(flatWeights) / dimsX

	var nb []float64
	for i := 0; i < len(layer.bias); i++ {
		nb = append(nb, layer.bias...)
	}

	ms := mat.NewDense(dimsX, dimsY, flatInputs)
	mw := mat.NewDense(dimsX, dimsY, flatWeights)
	mb := mat.NewDense(dimsX, dimsX, nb)
	output := (MatrixProduct(ms, mw))
	output.Add(output, mb)
	return output
}

func fromDense(dense *mat.Dense) [][]float64 {
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

// ix = range(N*j,N*(j+1))
// r = np.linspace(0.0,1,N) # radius
// t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
// X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
// y[ix] = j
func spiral_data(samples int, classes int) (*mat.Dense, *mat.Dense) {
	X := mat.NewDense(samples*classes, 2, nil)
	y := mat.NewDense(samples*classes, 1, nil)
	X.Zero()
	y.Zero()

	for class := 0; class < classes; class++ {
		for i := samples * class; i < samples*(class+1); i++ {
			r := float64(i) / float64(samples-1) // samples - 1
			t := float64(class*4)*(float64(i)*4)/float64(samples) + rand.NormFloat64()*0.2
			X.Set(i, 0, r*math.Sin(t*2.5))
			X.Set(i, 1, r*math.Cos(t*2.5))
			y.Set(i, 0, float64(class))
		}
	}

	return X, y
}

func RunModel() {
	inputs := [][]float64{
		{1, 2, 3, 2.5},
		{2.0, 5.0, -1.0, 2.0},
		{-1.5, 2.7, 3.3, -0.8},
	}

	layer1 := Layer{
		weights: [][]float64{
			{0.2, 0.8, -0.5, 1},
			{0.5, -0.91, 0.26, -0.5},
			{-0.26, -0.27, 0.17, 0.87},
		},
		bias: []float64{
			2.0, 3.0, 0.5,
		},
	}

	layer2 := Layer{
		weights: [][]float64{
			{0.1, -0.14, 0.5},
			{-0.5, 0.12, -0.33},
			{-0.44, 0.73, -0.13},
		},
		bias: []float64{
			-1, 2, -0.5,
		},
	}

	layer_1_output := runLayer(inputs, layer1)
	layer_2_output := runLayer(fromDense(layer_1_output), layer2)

	fmt.Println(mat.Formatted(layer_2_output))
}

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
	if err := p.Save(5*vg.Inch, 5*vg.Inch,
		time.Now().Local().Format(time.RFC3339)+"pointss.png"); err != nil {
		panic(err)
	}
}

func main() {
	classes := 3
	x, _ := spiral_data(100, classes)
	MakePlot(x, classes)
	return
}
