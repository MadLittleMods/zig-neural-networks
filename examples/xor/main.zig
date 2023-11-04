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

    try neural_network.NeuralNetwork.initFromLayerSizes(
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
        neural_network.CostFunction{
            .squared_error = {},
            // .cross_entropy = {},
        },
        allocator,
    );
    defer neural_network.deinitFromLayerSizes();
}
