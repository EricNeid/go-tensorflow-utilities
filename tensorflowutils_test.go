package tensorflowutils

import (
	"bytes"
	"io/ioutil"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNewModel(t *testing.T) {
	// action
	result, err := NewModel(
		"testdata/model/tensorflow_inception_graph.pb", "testdata/model/imagenet_comp_graph_label_strings.txt")
	// verify
	assert.NoError(t, err)
	assert.NotNil(t, result.GraphModel)
	assert.NotNil(t, result.SessionModel)
	assert.True(t, len(result.Labels) > 0)
}

func TestMakeTensorFromImage(t *testing.T) {
	// arrange
	image, _ := ioutil.ReadFile("testdata/balloon.jpeg")
	// action
	result, err := MakeTensorFromImage(bytes.NewBuffer(image), JPG)
	// verify
	assert.NoError(t, err)
	assert.NotNil(t, result)
}

func TestRun(t *testing.T) {
	// arrange
	model, _ := NewModel(
		"testdata/model/tensorflow_inception_graph.pb", "testdata/model/imagenet_comp_graph_label_strings.txt")
	image, _ := ioutil.ReadFile("testdata/balloon.jpeg")
	tensor, _ := MakeTensorFromImage(bytes.NewBuffer(image), JPG)
	// action
	result, err := model.Run(tensor)
	// verify
	assert.NoError(t, err)
	assert.True(t, len(result) > 0)
}

func TestClassifyImage(t *testing.T) {
	// arrange
	model, _ := NewModel(
		"testdata/model/tensorflow_inception_graph.pb", "testdata/model/imagenet_comp_graph_label_strings.txt")
	image, _ := ioutil.ReadFile("testdata/cat.jpeg")
	// action
	result, err := model.ClassifyImage(bytes.NewBuffer(image), JPG)
	// verify
	assert.NoError(t, err)
	assert.True(t, len(result) > 0)
}
