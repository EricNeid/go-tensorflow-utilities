package tensorflowutils

import "testing"
import "github.com/stretchr/testify/assert"

func TestLoadLabels(t *testing.T) {
	// action
	result, err := loadLabels("testdata/testlabels.txt")
	// verify
	assert.NoError(t, err)
	assert.EqualValues(t, []string{"Horse", "Car", "Bike", "Boat"}, result)
}
