const std = @import("std");
const log = std.log.scoped(.zig_neural_networks);

const NeuralNetwork = @import("../NeuralNetwork.zig");
const Layer = @import("../layers/Layer.zig");
const DenseLayer = @import("../layers/DenseLayer.zig");
const ActivationLayer = @import("../layers/ActivationLayer.zig");
const ActivationFunction = @import("../activation_functions.zig").ActivationFunction;
const CostFunction = @import("../cost_functions.zig").CostFunction;

// const CustomNoiseLayer = @import("CustomNoiseLayerForTesting");
const xor_data = @import("./data/xor_data.zig");

test "Serialize/deserialize standard neural network without errors" {
    const allocator = std.testing.allocator;

    // Neural network setup
    // =======================================
    //
    var neural_network = try NeuralNetwork.initFromLayerSizes(
        &[_]u32{ 2, 3, @typeInfo(xor_data.XorLabel).Enum.fields.len },
        ActivationFunction{ .sigmoid = .{} },
        ActivationFunction{ .soft_max = .{} },
        CostFunction{ .squared_error = .{} },
        allocator,
    );
    defer neural_network.deinit(allocator);

    // Test serialization/deserialization
    // =======================================
    //
    // Serialize the neural network
    const serialized_neural_network = try std.json.stringifyAlloc(
        allocator,
        neural_network,
        .{ .whitespace = .indent_2 },
    );
    defer allocator.free(serialized_neural_network);

    // Deserialize the neural network
    const parsed_neural_network = try std.json.parseFromSlice(
        NeuralNetwork,
        allocator,
        serialized_neural_network,
        .{},
    );
    defer parsed_neural_network.deinit();
    var deserialized_neural_network = parsed_neural_network.value;

    // These are arbitrary values that will give us an interesting/unique output to
    // compare. Basically just trying to avoid `0.0` and `1.0` since those are
    // straight-forward to get the right answer with XOR.
    const inputs = [_]f64{ 0.15, 0.85 };

    const expected_outputs = try neural_network.calculateOutputs(&inputs, allocator);
    defer allocator.free(expected_outputs);

    const actual_outputs = try deserialized_neural_network.calculateOutputs(&inputs, allocator);
    defer allocator.free(actual_outputs);

    // Check to make sure the neural network outputs are the same after
    // serialization/deserialization
    try std.testing.expectEqualSlices(
        f64,
        expected_outputs,
        actual_outputs,
    );
}

// test "Serialize/deserialize neural network with custom layer types" {
//     const allocator = std.testing.allocator;

//     // Neural network setup
//     // =======================================
//     //
//     // Register the custom layer types we will be using with the library (this is used
//     // for deserialization).
//     try Layer.registerCustomLayer(CustomNoiseLayer, allocator);
//     defer Layer.deinitCustomLayerMap(allocator);

//     // Setup the layers we'll be using in our custom neural network.
//     // We just choose a variety of layer sizes and activation functions to test here
//     var custom_noise_layer = try CustomNoiseLayer.init(
//         0.01,
//         0.75,
//         // It's nicer to have a fixed seed so we can reproduce the same results.
//         123,
//     );
//     var dense_layer1 = try DenseLayer.init(2, 3, allocator);
//     var activation_layer1 = try ActivationLayer.init(ActivationFunction{ .elu = .{} });
//     var dense_layer2 = try DenseLayer.init(3, 3, allocator);
//     var activation_layer2 = try ActivationLayer.init(ActivationFunction{ .elu = .{} });
//     var dense_layer3 = try DenseLayer.init(
//         3,
//         @typeInfo(xor_data.XorLabel).Enum.fields.len,
//         allocator,
//     );
//     var activation_layer3 = try ActivationLayer.init(ActivationFunction{ .soft_max = .{} });

//     var layers = [_]Layer{
//         custom_noise_layer.layer(),
//         dense_layer1.layer(),
//         activation_layer1.layer(),
//         dense_layer2.layer(),
//         activation_layer2.layer(),
//         dense_layer3.layer(),
//         activation_layer3.layer(),
//     };
//     defer for (&layers) |*layer| {
//         layer.deinit(allocator);
//     };

//     var neural_network = try NeuralNetwork.initFromLayers(
//         &layers,
//         CostFunction{ .squared_error = .{} },
//     );

//     // Test serialization/deserialization
//     // =======================================
//     //
//     // Serialize the neural network
//     const serialized_neural_network = try std.json.stringifyAlloc(
//         allocator,
//         neural_network,
//         .{ .whitespace = .indent_2 },
//     );
//     defer allocator.free(serialized_neural_network);

//     // Deserialize the neural network
//     const parsed_neural_network = try std.json.parseFromSlice(
//         NeuralNetwork,
//         allocator,
//         serialized_neural_network,
//         .{},
//     );
//     defer parsed_neural_network.deinit();
//     var deserialized_neural_network = parsed_neural_network.value;

//     // These are arbitrary values that will give us an interesting/unique output to
//     // compare. Basically just trying to avoid `0.0` and `1.0` since those are
//     // straight-forward to get the right answer with XOR.
//     const inputs = [_]f64{ 0.15, 0.85 };

//     const expected_outputs = try neural_network.calculateOutputs(&inputs, allocator);
//     defer allocator.free(expected_outputs);

//     const actual_outputs = try deserialized_neural_network.calculateOutputs(&inputs, allocator);
//     defer allocator.free(actual_outputs);

//     // Check to make sure the neural network outputs are the same after
//     // serialization/deserialization
//     try std.testing.expectEqualSlices(
//         f64,
//         expected_outputs,
//         actual_outputs,
//     );
// }
