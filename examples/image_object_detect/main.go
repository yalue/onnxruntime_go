package main

import (
	"bytes"
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"math"
	"os"
	"sort"
	"time"

	"github.com/nfnt/resize"
	ort "github.com/yalue/onnxruntime_go"
)

// Embed the model file into the binary
var modelPath = "./models/yolov8n.onnx"

// Embed the libonnxruntime shared library into the binary
var libPath = "../../lib/osx/libonnxruntime.1.15.1.dylib"
var imagePath = "./car.png"
var useCoreML = true

var sess *ort.Session[float32]
var inputT *ort.Tensor[float32]
var outputT *ort.Tensor[float32]
var blank []float32

type ModelSession struct {
	Session *ort.Session[float32]
	Input   *ort.Tensor[float32]
	Output  *ort.Tensor[float32]
}

func main() {

	// Open the image file
	file, err := os.Open(imagePath)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	// Read the entire file into memory
	imageData, err := io.ReadAll(file)
	if err != nil {
		panic(err)
	}

	// Create a bytes.Buffer from the imageData
	imageBuffer := bytes.NewBuffer(imageData)

	// Prep blank
	reader := bytes.NewReader(imageBuffer.Bytes())
	blank, _, _ = prepare_input(reader)

	modelSes := ModelSession{}
	if useCoreML {
		modelSes, err = initSessionCoreML()
		if err != nil {
			panic(err)
		}
	} else {
		modelSes, err = initSessionCPU()
		if err != nil {
			panic(err)
		}
	}

	// Run the detection 5 times
	for i := 0; i < 5; i++ {
		println("Run", i+1, ":")

		// Create a new reader from the buffer for each iteration
		reader := bytes.NewReader(imageBuffer.Bytes())
		input, img_width, img_height := prepare_input(reader)

		timer := time.Now()
		output, err := runInference(modelSes, input)
		if err != nil {
			panic(err)
		}
		fmt.Printf("TOOK: %dms\n", time.Since(timer).Milliseconds())

		// Print execution time
		boxes := process_output(output, img_width, img_height)

		// Print the results
		for _, box := range boxes {
			objectName := box[4].(string)  // Accessing the object name
			confidence := box[5].(float32) // Accessing the confidence
			x1 := box[0].(float64)         // Accessing the x1 coordinate
			y1 := box[1].(float64)         // Accessing the y1 coordinate
			x2 := box[2].(float64)         // Accessing the x2 coordinate
			y2 := box[3].(float64)         // Accessing the y2 coordinate
			fmt.Printf("Object: %s Confidence: %.2f Coordinates: (%f, %f), (%f, %f)\n", objectName, confidence, x1, y1, x2, y2)
		}
		println()
	}
}

func prepare_input(buffer io.Reader) ([]float32, int64, int64) {
	// Decode the image from the buffer
	imageObj, _, _ := image.Decode(buffer)

	// Get the image size
	imageSize := imageObj.Bounds().Size()
	imageWidth, imageHeight := int64(imageSize.X), int64(imageSize.Y)

	// Resize the image to 640x640 using Lanczos3 algorithm
	imageObj = resize.Resize(640, 640, imageObj, resize.Lanczos3)

	// Initialize slices to store red, green, blue channels
	redChannel := []float32{}
	greenChannel := []float32{}
	blueChannel := []float32{}

	// Iterate through pixels and populate the channel slices
	for y := 0; y < 640; y++ {
		for x := 0; x < 640; x++ {
			r, g, b, _ := imageObj.At(x, y).RGBA()
			redChannel = append(redChannel, float32(r/257)/255.0)
			greenChannel = append(greenChannel, float32(g/257)/255.0)
			blueChannel = append(blueChannel, float32(b/257)/255.0)
		}
	}

	// Concatenate the channel slices to create the final input
	inputArray := append(redChannel, greenChannel...)
	inputArray = append(inputArray, blueChannel...)

	return inputArray, imageWidth, imageHeight
}

func runInference(modelSes ModelSession, input []float32) ([]float32, error) {
	inTensor := modelSes.Input.GetData()
	copy(inTensor, input)
	err := modelSes.Session.Run()
	if err != nil {
		return nil, err
	}
	return modelSes.Output.GetData(), nil
}

func initSessionCPU() (ModelSession, error) {
	ort.SetSharedLibraryPath(libPath)
	err := ort.InitializeEnvironment()
	if err != nil {
		return ModelSession{}, err
	}

	inputShape := ort.NewShape(1, 3, 640, 640)
	inputTensor, err := ort.NewTensor(inputShape, blank)
	if err != nil {
		return ModelSession{}, err
	}

	outputShape := ort.NewShape(1, 84, 8400)
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return ModelSession{}, err
	}

	session, err := ort.NewSession[float32](modelPath,
		[]string{"images"}, []string{"output0"},
		[]*ort.Tensor[float32]{inputTensor}, []*ort.Tensor[float32]{outputTensor})

	modelSes := ModelSession{
		Session: session,
		Input:   inputTensor,
		Output:  outputTensor,
	}

	return modelSes, err
}

func initSessionCoreML() (ModelSession, error) {
	// Set the shared library path and initialize the environment
	ort.SetSharedLibraryPath(libPath)
	err := ort.InitializeEnvironment()
	if err != nil {
		return ModelSession{}, err
	}

	// Define the input and output shapes and tensors
	inputShape := ort.NewShape(1, 3, 640, 640)
	inputTensor, err := ort.NewTensor(inputShape, blank)
	if err != nil {
		return ModelSession{}, err
	}
	outputShape := ort.NewShape(1, 84, 8400)
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return ModelSession{}, err
	}

	// Define input and output names
	inputNames := []string{"images"}
	outputNames := []string{"output0"}

	// Create a session with CoreML Execution Provider
	coremlFlags := uint32(0) // No specific CoreML flags
	session, err := ort.CreateSessionWithCoreML[float32](modelPath, coremlFlags, inputNames, outputNames,
		[]*ort.Tensor[float32]{inputTensor}, []*ort.Tensor[float32]{outputTensor})
	if err != nil {
		return ModelSession{}, err
	}

	// Create the ModelSession object
	modelSes := ModelSession{
		Session: session,
		Input:   inputTensor,
		Output:  outputTensor,
	}

	return modelSes, nil
}

func process_output(output []float32, imgWidth, imgHeight int64) [][]interface{} {
	// Define a slice to hold the bounding boxes
	boundingBoxes := [][]interface{}{}

	// Iterate through the output array, considering 8400 indices
	for idx := 0; idx < 8400; idx++ {
		classID, probability := 0, float32(0.0)
		// Iterate through 80 classes and find the class with the highest probability
		for col := 0; col < 80; col++ {
			currentProb := output[8400*(col+4)+idx]
			if currentProb > probability {
				probability = currentProb
				classID = col
			}
		}

		// If the probability is less than 0.5, continue to the next index
		if probability < 0.5 {
			continue
		}

		// Retrieve the label associated with the class ID
		label := yolo_classes[classID]

		// Extract the coordinates and dimensions of the bounding box
		xc, yc, w, h := output[idx], output[8400+idx], output[2*8400+idx], output[3*8400+idx]
		x1 := (xc - w/2) / 640 * float32(imgWidth)
		y1 := (yc - h/2) / 640 * float32(imgHeight)
		x2 := (xc + w/2) / 640 * float32(imgWidth)
		y2 := (yc + h/2) / 640 * float32(imgHeight)

		// Append the bounding box to the result
		boundingBoxes = append(boundingBoxes, []interface{}{float64(x1), float64(y1), float64(x2), float64(y2), label, probability})
	}

	// Sort the bounding boxes by probability
	sort.Slice(boundingBoxes, func(i, j int) bool {
		return boundingBoxes[i][5].(float32) < boundingBoxes[j][5].(float32)
	})

	// Define a slice to hold the final result
	result := [][]interface{}{}

	// Iterate through sorted bounding boxes, removing overlaps
	for len(boundingBoxes) > 0 {
		result = append(result, boundingBoxes[0])
		tmp := [][]interface{}{}
		for _, box := range boundingBoxes {
			if iou(boundingBoxes[0], box) < 0.7 {
				tmp = append(tmp, box)
			}
		}
		boundingBoxes = tmp
	}

	return result
}

func iou(box1, box2 []interface{}) float64 {
	// Calculate the area of intersection between the two bounding boxes using the intersection function
	intersectArea := intersection(box1, box2)

	// Calculate the union of the two bounding boxes using the union function
	unionArea := union(box1, box2)

	// The Intersection over Union (IoU) is the ratio of the intersection area to the union area
	return intersectArea / unionArea
}

func union(box1, box2 []interface{}) float64 {
	// Extract coordinates of the first rectangle
	rect1Left, rect1Bottom, rect1Right, rect1Top := box1[0].(float64), box1[1].(float64), box1[2].(float64), box1[3].(float64)

	// Extract coordinates of the second rectangle
	rect2Left, rect2Bottom, rect2Right, rect2Top := box2[0].(float64), box2[1].(float64), box2[2].(float64), box2[3].(float64)

	// Calculate area of the first rectangle
	rect1Area := (rect1Right - rect1Left) * (rect1Top - rect1Bottom)

	// Calculate area of the second rectangle
	rect2Area := (rect2Right - rect2Left) * (rect2Top - rect2Bottom)

	// Use the intersection function to calculate the area of overlap between the two rectangles
	intersectArea := intersection(box1, box2)

	// The union of two rectangles is the sum of their areas minus the area of their overlap
	return rect1Area + rect2Area - intersectArea
}

func intersection(box1, box2 []interface{}) float64 {
	// Extracting the coordinates of the first box
	firstBoxX1, firstBoxY1, firstBoxX2, firstBoxY2 := box1[0].(float64), box1[1].(float64), box1[2].(float64), box1[3].(float64)

	// Extracting the coordinates of the second box
	secondBoxX1, secondBoxY1, secondBoxX2, secondBoxY2 := box2[0].(float64), box2[1].(float64), box2[2].(float64), box2[3].(float64)

	// Calculating the x coordinate of the left side of the intersection
	intersectX1 := math.Max(firstBoxX1, secondBoxX1)

	// Calculating the y coordinate of the bottom side of the intersection
	intersectY1 := math.Max(firstBoxY1, secondBoxY1)

	// Calculating the x coordinate of the right side of the intersection
	intersectX2 := math.Min(firstBoxX2, secondBoxX2)

	// Calculating the y coordinate of the top side of the intersection
	intersectY2 := math.Min(firstBoxY2, secondBoxY2)

	// Calculating and returning the area of the intersection
	return (intersectX2 - intersectX1) * (intersectY2 - intersectY1)
}

// Array of YOLOv8 class labels
var yolo_classes = []string{
	"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
	"traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
	"sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
	"suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
	"skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
	"bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
	"cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
	"remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
	"clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
}
