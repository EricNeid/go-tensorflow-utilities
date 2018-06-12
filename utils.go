package tensorflowutils

import (
	"bufio"
	"errors"
	"io/ioutil"
	"os"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

// loadLabels loads labels from the given file, assuming each label on separate line.
func loadLabels(labelFile string) ([]string, error) {
	labelsFile, err := os.Open(labelFile)
	if err != nil {
		return nil, err
	}
	defer labelsFile.Close()
	scanner := bufio.NewScanner(labelsFile)

	// assuming labels are separated by newlines
	var labels []string
	for scanner.Scan() {
		labels = append(labels, scanner.Text())
	}
	if scanner.Err() != nil {
		return nil, err
	}
	return labels, nil
}

func loadGraphModel(modelFile string) (*tf.Graph, *tf.Session, error) {
	// load model
	model, err := ioutil.ReadFile(modelFile)
	if err != nil {
		return nil, nil, err
	}
	graphModel := tf.NewGraph()
	if err := graphModel.Import(model, ""); err != nil {
		return nil, nil, err
	}
	// create session
	sessionModel, err := tf.NewSession(graphModel, nil)
	if err != nil {
		return nil, nil, err
	}
	return graphModel, sessionModel, nil
}

// makeTransformImageGraph creates a graph to decode, rezise and normalize an image.
func makeTransformImageGraph(imageFormat ImageType) (graph *tf.Graph, input, output tf.Output, err error) {
	const (
		H, W  = 224, 224
		Mean  = float32(117)
		Scale = float32(1)
	)
	s := op.NewScope()
	input = op.Placeholder(s, tf.String)

	// Decode PNG or JPEG
	var decode tf.Output
	if imageFormat == PNG {
		decode = op.DecodePng(s, input, op.DecodePngChannels(3))
	} else if imageFormat == JPG {
		decode = op.DecodeJpeg(s, input, op.DecodeJpegChannels(3))
	} else {
		err := errors.New("Unsupported image type given: " + imageFormat.string() + " Expecting " + PNG.string() + " or " + JPG.string())
		return nil, input, decode, err
	}

	// Div and Sub perform (value-Mean)/Scale for each pixel
	output = op.Div(s,
		op.Sub(s,
			// Resize to 224x224 with bilinear interpolation
			op.ResizeBilinear(s,
				// Create a batch containing a single image
				op.ExpandDims(s,
					// Use decoded pixel values
					op.Cast(s, decode, tf.Float),
					op.Const(s.SubScope("make_batch"), int32(0))),
				op.Const(s.SubScope("size"), []int32{H, W})),
			op.Const(s.SubScope("mean"), Mean)),
		op.Const(s.SubScope("scale"), Scale))
	graph, err = s.Finalize()
	return graph, input, output, err
}

func (imageType ImageType) string() string {
	return string(imageType)
}
