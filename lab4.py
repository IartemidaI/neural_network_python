import numpy
import time
def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))
training_inputs = numpy.array([[0.1, 0.2, 0.3, 0.4, 0.5], [0.2, 0.3, 0.4, 0.5, 0.6], [0.4, 0.5, 0.6, 0.7, 0.8], [0.9, 0.7, 0.8, 0.6, 0.6]])
training_outputs = numpy.array([[0.5, 0.6, 0.8, 0.9]]).T
synaptic_weights = 2*numpy.random.random((5, 1)) - 1
print("Start synaptic weights:\n", synaptic_weights)
start_time=time.time()
f = open("log.txt", "w")
for i in range(40000):
    input_layer = training_inputs
    outputs = sigmoid(numpy.dot(input_layer, synaptic_weights))
    error = training_outputs - outputs
    adjustment = numpy.dot(input_layer.T, error * (outputs* (1-outputs)))
    for i in adjustment:
        f.write(str(round(i[0], 8)) + "\t")
    f.write("\n")
    synaptic_weights = synaptic_weights + adjustment
print("Training time: {:.3f} seconds".format(time.time()-start_time))
f.close()
print("Training result:\n", outputs)
new_input = numpy.array([0.1, 0.1, 0.3, 0.1, 0.2])
output = sigmoid(numpy.dot(new_input, synaptic_weights))
print("New output:", output)