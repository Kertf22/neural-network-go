package main

import (
	// "fmt"
	"fmt"
	"image/color"

	"neural-network-go/activations"
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

func RunModel() {
	classes := 3
	x, y := dataset.SpiralData(100, classes)
	MakePlot(x, classes)
	//
	l := layer.Layer{}
	l.InitLayer(2, 3)

	output := l.Foward(x)
	activations.ReLu(output)

	l2 := layer.Layer{}
	l2.InitLayer(3, 3)

	output = l2.Foward(output)
	activations.SoftMax(output)

	lossValue := loss.LossClassification(output, y)
	fmt.Println(lossValue)
}

func main() {

	RunModel()
	return
}
