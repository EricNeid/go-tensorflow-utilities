package tensorflowutils

import tf "github.com/tensorflow/tensorflow/tensorflow/go"

// Model represents a loaded tensorflow graph model with it's labels
// an the generated sessionModel for this graph.
type Model struct {
	model *tf.Graph,
	labels []string,
	sessionModel *tf.Session
}

func NewModel(modelFile string, lableFile string) (*Model, err) {
	// load model
	model, err := ioutil.ReadFile(modelFile)
	if err != nil {
		return nil, err
	}
	graphModel = tf.NewGraph()
	if err := graphModel.Import(model, ""); err != nil {
		return nil, err
	}

	// create session
	sessionModel, err = tf.NewSession(graphModel, nil)
	if err != nil {
		return nil, err
	}

	// load labels
	labelsFile, err := os.Open(lableFile)
	if err != nil {
		return nil, err
	}
	defer labelsFile.Close()
	scanner := bufio.NewScanner(labelsFile)

	// assuming labels are separated by newlines
	for scanner.Scan() {
		labels = append(labels, scanner.Text())
	}
	if scanner.Err() != nil {
		return nil, err
	}

	return Model {

	}
}
