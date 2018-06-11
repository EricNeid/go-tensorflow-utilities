package tensorflowutils

import (
	"bufio"
	"io/ioutil"
	"log"
	"os"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

// Model represents a loaded tensorflow graph model with it's labels
// an the generated sessionModel for this graph.
type Model struct {
	sessionModel *tf.Session
	graphModel   *tf.Graph
	labels       []string
}

// NewModel loads graphModel and label from given filepath and returns a new
// Model, containing the Graph, it's labels and the session.
// It is assumed that the labels are separated by newlines.
func NewModel(modelFile string, lableFile string) (*Model, error) {
	graphModel, sessionModel, err := loadGraphModel(modelFile)
	if err != nil {
		log.Println("Error while loading model.")
		return nil, err
	}

	labels, err := loadLabels(lableFile)
	if err != nil {
		log.Println("Error while loading labels.")
		return nil, err
	}

	return &Model{
		sessionModel: sessionModel,
		graphModel:   graphModel,
		labels:       labels,
	}, nil
}

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
