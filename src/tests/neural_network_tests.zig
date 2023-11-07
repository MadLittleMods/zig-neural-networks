const std = @import("std");
const log = std.log.scoped(.zig_neural_networks);

const DataPoint = @import("../data_point.zig").DataPoint;
const NeuralNetwork = @import("../neural_network.zig").NeuralNetwork;
const Layer = @import("../layers/layer.zig").Layer;
const DenseLayer = @import("../layers/dense_layer.zig").DenseLayer;
const ActivationLayer = @import("../layers/activation_layer.zig").ActivationLayer;
const ActivationFunction = @import("../activation_functions.zig").ActivationFunction;
const CostFunction = @import("../cost_functions.zig").CostFunction;

const cost_gradient_utils = @import("./utils/cost_gradient_utils.zig");

// Binary value can only be 0 or 1
const xor_labels = [_]u8{
    0,
    1,
};
const XorDataPoint = DataPoint(u8, &xor_labels);
// The XOR data points
var xor_data_points = [_]XorDataPoint{
    XorDataPoint.init(&[_]f64{ 0, 0 }, 0),
    XorDataPoint.init(&[_]f64{ 0, 1 }, 1),
    XorDataPoint.init(&[_]f64{ 1, 0 }, 1),
    XorDataPoint.init(&[_]f64{ 1, 1 }, 0),
};

test "Gradient check various layers with the most basic XOR dataset" {
    const allocator = std.testing.allocator;

    // We just choose a variety of layer sizes and activation functions to test here
    var dense_layer1 = try DenseLayer.init(2, 3, allocator);
    var activation_layer1 = try ActivationLayer.init(ActivationFunction{ .sigmoid = .{} });
    var dense_layer2 = try DenseLayer.init(3, 3, allocator);
    var activation_layer2 = try ActivationLayer.init(ActivationFunction{ .elu = .{} });
    var dense_layer3 = try DenseLayer.init(3, xor_labels.len, allocator);
    // Testing SoftMax is of particular interest because it's the only multi-input activation function
    var activation_layer3 = try ActivationLayer.init(ActivationFunction{ .soft_max = .{} });

    // Keep track of the dense layers to do gradient checks against
    var dense_layers = [_]DenseLayer{
        dense_layer1,
        dense_layer2,
        dense_layer3,
    };

    var layers = [_]Layer{
        dense_layer1.layer(),
        activation_layer1.layer(),
        dense_layer2.layer(),
        activation_layer2.layer(),
        dense_layer3.layer(),
        activation_layer3.layer(),
    };
    defer {
        for (&layers) |*layer| {
            layer.deinit(allocator);
        }
    }

    var neural_network = try NeuralNetwork(XorDataPoint).initFromLayers(
        &layers,
        CostFunction{ .squared_error = .{} },
    );

    const learn_rate: f64 = 0.1;

    var current_epoch_index: usize = 0;
    while (current_epoch_index < 15) : (current_epoch_index += 1) {
        // Check all of the layers at various points during the training process to make
        // sure the cost gradients are correct.
        //
        // This is equivalent to a `learn(...)` step but allows us to introspect the
        // cost gradients in the layers before the network applies them.
        if (current_epoch_index == 0 or
            current_epoch_index == 1 or
            current_epoch_index == 7 or
            current_epoch_index == 11 or
            current_epoch_index == 14 or
            current_epoch_index == 15)
        {
            // Use the backpropagation algorithm to calculate the gradient of the cost function
            // (with respect to the network's weights and biases). This is done for each data point,
            // and the gradients are added together. (assembling the analytical gradient)
            try neural_network._updateCostGradients(
                &xor_data_points,
                allocator,
            );

            // Do our gradient checks!
            //
            // We only do this for the dense layers because the activation layers are
            // just pass-through layers that don't have any weights or biases to update
            // (no network parameters).
            for (&dense_layers) |*dense_layer| {
                try cost_gradient_utils.sanityCheckCostGradients(
                    @TypeOf(neural_network),
                    &neural_network,
                    dense_layer,
                    &xor_data_points,
                    allocator,
                );
            }

            // Gradient descent step: update all weights and biases in the network
            try neural_network._applyCostGradients(
                learn_rate,
                0,
                xor_data_points.len,
            );
        }
        // Do a standard learning step
        else {
            try neural_network.learn(
                &xor_data_points,
                // TODO: Implement learn rate decay so we take more refined steps the
                // longer we train for.
                learn_rate,
                0,
                allocator,
            );
        }
    }
}
