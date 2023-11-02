const std = @import("std");
const neural_network = @import("zig-neural-network");

// Set the logging levels
pub const std_options = struct {
    pub const log_level = .debug;

    pub const log_scope_levels = &[_]std.log.ScopeLevel{
        .{ .scope = .zig_neural_network, .level = .debug },
    };
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer {
        switch (gpa.deinit()) {
            .ok => {},
            .leak => std.log.err("GPA allocator: Memory leak detected", .{}),
        }
    }

    // I wish we could just inline these declarations as literals in the `layers` array
    // so there wasn't a chance to re-assemble in the wrong order or accidenately use
    // layers multiple times. But Zig will create any literal/temporary declaration as
    // `const` with no way to specify that they are `var`/mutable
    // (https://ziglang.org/download/0.10.0/release-notes.html#Address-of-Temporaries-Now-Produces-Const-Pointers).
    // And we would also have trouble with deinitializing them since we wouldn't have a
    // handle to them.
    // var dense_layer1 = neural_network.DenseLayer.init(2, 3, allocator);
    // defer dense_layer1.deinit();
    // var activation_layer1 = neural_network.ActivationLayer(neural_network.ActivationFunction{ .elu = {} }).init();
    // var dense_layer2 = neural_network.DenseLayer.init(3, 2);
    // defer dense_layer2.deinit();
    // var activation_layer2 = neural_network.ActivationLayer(neural_network.ActivationFunction{ .soft_max = {} }).init();

    // var layers = [_]neural_network.Layer{
    //     dense_layer1.layer(),
    //     activation_layer1.layer(),
    //     dense_layer2.layer(),
    //     activation_layer2.layer(),
    // };

    neural_network.NeuralNetwork.initFromLayerSizes(
        &[_]u32{ 2, 3, 2 },
        neural_network.ActivationFunction{
            // .relu = .{},
            // .leaky_relu = .{},
            .elu = .{},
            // .sigmoid = .{},
        },
        neural_network.ActivationFunction{
            .soft_max = .{},
            // .sigmoid = .{},
        },
        neural_network.LossFunction{
            .squared_error = {},
            // .cross_entropy = {},
        },
        allocator,
    );

    std.log.debug("layers {any}", .{layers});
}
