const std = @import("std");
const neural_networks = @import("zig-neural-networks");
const CustomNoiseLayer = @import("CustomNoiseLayer.zig");

const mnist_main = @import("main.zig");

const BATCH_SIZE: u32 = 100;
const LEARN_RATE: f64 = 0.05;
const MOMENTUM = 0.9;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer {
        switch (gpa.deinit()) {
            .ok => {},
            .leak => std.log.err("GPA allocator: Memory leak detected", .{}),
        }
    }

    const start_timestamp_seconds = std.time.timestamp();
    _ = start_timestamp_seconds;

    // Getting the training/testing data ready
    // =======================================
    //
    const parsed_mnist_data = try mnist_main.getMnistDataPoints(allocator);
    defer parsed_mnist_data.deinit();
    const mnist_data = parsed_mnist_data.value;

    // Neural network
    // =======================================
    //
    var custom_noise_layer = try CustomNoiseLayer.init(0.01, 0.75);
    var dense_layer1 = try neural_networks.DenseLayer.init(784, 100, allocator);
    var activation_layer1 = try neural_networks.ActivationLayer.init(neural_networks.ActivationFunction{
        .elu = .{},
    });
    var dense_layer2 = try neural_networks.DenseLayer.init(100, @typeInfo(mnist_main.DigitLabel).Enum.fields.len, allocator);
    var activation_layer2 = try neural_networks.ActivationLayer.init(neural_networks.ActivationFunction{
        .soft_max = .{},
    });

    var base_layers = [_]neural_networks.Layer{
        dense_layer1.layer(),
        activation_layer1.layer(),
        dense_layer2.layer(),
        activation_layer2.layer(),
    };
    var training_layers = [_]neural_networks.Layer{
        // The CustomNoiseLayer should only used during training to reduce overfitting.
        // It doesn't make sense to run during testing because we don't want to skew our
        // inputs at all.
        custom_noise_layer.layer(),
    } ++ base_layers;
    defer {
        for (&training_layers) |*layer| {
            layer.deinit(allocator);
        }
    }

    var neural_network_for_training = try neural_networks.NeuralNetwork.initFromLayers(
        &training_layers,
        neural_networks.CostFunction{ .cross_entropy = .{} },
    );
    defer neural_network_for_training.deinit(allocator);

    var neural_network_for_testing = try neural_networks.NeuralNetwork.initFromLayers(
        &base_layers,
        neural_networks.CostFunction{ .cross_entropy = .{} },
    );
    defer neural_network_for_testing.deinit(allocator);

    try mnist_main.train(
        &neural_network_for_training,
        &neural_network_for_testing,
        mnist_data,
        allocator,
    );
}
