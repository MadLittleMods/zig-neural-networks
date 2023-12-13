const std = @import("std");
const log = std.log.scoped(.zig_neural_networks);

const DataPoint = @import("../data_point.zig").DataPoint;
const NeuralNetwork = @import("../NeuralNetwork.zig");
const Layer = @import("../layers/Layer.zig");
const DenseLayer = @import("../layers/DenseLayer.zig");
const ActivationLayer = @import("../layers/ActivationLayer.zig");
const ActivationFunction = @import("../activation_functions.zig").ActivationFunction;
const CostFunction = @import("../cost_functions.zig").CostFunction;

const cost_gradient_utils = @import("./utils/cost_gradient_utils.zig");

const xor_data = @import("./data/xor_data.zig");
const iris_flower_data = @import("./data/iris_flower_data.zig");

fn gradientCheckNeuralNetwork(
    neural_network: *NeuralNetwork,
    dense_layers: []DenseLayer,
    training_data_batch: []const DataPoint,
) !void {
    const allocator = std.testing.allocator;
    const learn_rate: f64 = 0.1;

    var current_epoch_index: usize = 0;
    while (current_epoch_index < 8) : (current_epoch_index += 1) {
        // Check all of the layers at various *arbitrary* points during the training
        // process to make sure the cost gradients are correct.
        //
        // This is equivalent to a `learn(...)` step but allows us to introspect the
        // cost gradients in the layers before the network applies them.
        if (current_epoch_index == 0 or
            current_epoch_index == 1 or
            current_epoch_index == 2 or
            current_epoch_index == 5 or
            current_epoch_index == 7 or
            current_epoch_index == 8)
        {
            // Use the backpropagation algorithm to calculate the gradient of the cost function
            // (with respect to the network's weights and biases). This is done for each data point,
            // and the gradients are added together. (assembling the analytical gradient)
            try neural_network._updateCostGradients(
                training_data_batch,
                allocator,
            );

            // Do our gradient checks!
            //
            // We only do this for the dense layers because the activation layers are
            // just pass-through layers that don't have any weights or biases to update
            // (no network parameters).
            for (dense_layers) |*dense_layer| {
                try cost_gradient_utils.sanityCheckCostGradients(
                    neural_network,
                    dense_layer,
                    training_data_batch,
                    allocator,
                );
            }

            // Gradient descent step: update all weights and biases in the network
            try neural_network._applyCostGradients(
                learn_rate,
                0,
                training_data_batch.len,
            );
        }
        // Do a standard learning step
        else {
            try neural_network.learn(
                training_data_batch,
                learn_rate,
                0,
                allocator,
            );
        }
    }
}

test "Gradient check various layers with the most basic XOR dataset (2 inputs, 2 ouputs)" {
    const allocator = std.testing.allocator;

    // We just choose a variety of layer sizes and activation functions to test here
    var dense_layer1 = try DenseLayer.init(2, 3, allocator);
    var activation_layer1 = try ActivationLayer.init(ActivationFunction{ .sigmoid = .{} });
    var dense_layer2 = try DenseLayer.init(3, 3, allocator);
    var activation_layer2 = try ActivationLayer.init(ActivationFunction{ .elu = .{} });
    var dense_layer3 = try DenseLayer.init(
        3,
        @typeInfo(xor_data.XorLabel).Enum.fields.len,
        allocator,
    );
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
    defer for (&layers) |*layer| {
        layer.deinit(allocator);
    };

    var neural_network = try NeuralNetwork.initFromLayers(
        &layers,
        CostFunction{ .squared_error = .{} },
    );

    try gradientCheckNeuralNetwork(
        &neural_network,
        &dense_layers,
        &xor_data.xor_data_points,
    );
}

test "Gradient check various layers with more inputs/outputs" {
    const allocator = std.testing.allocator;

    // We just choose a variety of layer sizes and activation functions to test here
    var dense_layer1 = try DenseLayer.init(4, 8, allocator);
    var activation_layer1 = try ActivationLayer.init(ActivationFunction{ .sigmoid = .{} });
    var dense_layer2 = try DenseLayer.init(8, 6, allocator);
    var activation_layer2 = try ActivationLayer.init(ActivationFunction{ .elu = .{} });
    var dense_layer3 = try DenseLayer.init(
        6,
        @typeInfo(iris_flower_data.IrisFlowerLabel).Enum.fields.len,
        allocator,
    );
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
    defer for (&layers) |*layer| {
        layer.deinit(allocator);
    };

    var neural_network = try NeuralNetwork.initFromLayers(
        &layers,
        CostFunction{ .cross_entropy = .{} },
    );

    try gradientCheckNeuralNetwork(
        &neural_network,
        &dense_layers,
        // Just use a subset of the data points to speed up the test
        iris_flower_data.iris_flower_data_points[0..50],
    );
}
