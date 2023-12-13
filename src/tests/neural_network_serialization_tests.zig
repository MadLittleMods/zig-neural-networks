const std = @import("std");
const log = std.log.scoped(.zig_neural_networks);

const NeuralNetwork = @import("../NeuralNetwork.zig");
const ActivationFunction = @import("../activation_functions.zig").ActivationFunction;
const CostFunction = @import("../cost_functions.zig").CostFunction;

const xor_data = @import("./data/xor_data.zig");

test "Serialize/deserialize standard neural network without errors" {
    const allocator = std.testing.allocator;
    var neural_network = try NeuralNetwork.initFromLayerSizes(
        &[_]u32{ 2, 3, @typeInfo(xor_data.XorLabel).Enum.fields.len },
        ActivationFunction{ .sigmoid = .{} },
        ActivationFunction{ .soft_max = .{} },
        CostFunction{ .squared_error = .{} },
        allocator,
    );
    defer neural_network.deinit(allocator);

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

// TODO: Add a test for a custom Layer type
