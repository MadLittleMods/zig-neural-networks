const std = @import("std");
const neural_networks = @import("zig-neural-networks");

const mnist_main = @import("main.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer {
        switch (gpa.deinit()) {
            .ok => {},
            .leak => std.log.err("GPA allocator: Memory leak detected", .{}),
        }
    }

    // Getting the training/testing data ready
    // =======================================
    //
    const parsed_mnist_data = try mnist_main.getMnistDataPoints(allocator);
    defer parsed_mnist_data.deinit();
    const mnist_data = parsed_mnist_data.value;

    // TODO: Do neural network stuff here
    std.log.debug("Training data: {}", .{mnist_data.training_data_points.len});

    var dense_layer1 = try neural_networks.DenseLayer.init(2, 3, allocator);
    var activation_layer1 = try neural_networks.ActivationLayer.init(neural_networks.ActivationFunction{ .sigmoid = .{} });
    var dense_layer2 = try neural_networks.DenseLayer.init(3, 3, allocator);
    var activation_layer2 = try neural_networks.ActivationLayer.init(neural_networks.ActivationFunction{ .elu = .{} });
    var dense_layer3 = try neural_networks.DenseLayer.init(3, 2, allocator);
    var activation_layer3 = try neural_networks.ActivationLayer.init(neural_networks.ActivationFunction{ .soft_max = .{} });

    var layers = [_]neural_networks.Layer{
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

    var neural_network = try neural_networks.NeuralNetwork.initFromLayers(
        &layers,
        neural_networks.CostFunction{ .squared_error = .{} },
    );
    _ = neural_network;
}
