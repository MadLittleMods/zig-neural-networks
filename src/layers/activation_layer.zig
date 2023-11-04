const std = @import("std");
const log = std.log.scoped(.zig_neural_network);

const Layer = @import("layer.zig").Layer;
const ApplyCostGradientsOptions = @import("layer.zig").ApplyCostGradientsOptions;
const ActivationFunction = @import("../activation_functions.zig").ActivationFunction;

/// A layer that applies an activation function to its inputs.
///
/// (inherits from `Layer`)
pub fn ActivationLayer(activation_function: ActivationFunction) type {
    return struct {
        const Self = @This();
        /// Store any inputs we get during the forward pass so we can use them during
        /// the backward pass.
        inputs: []f64 = undefined,

        pub fn init() !Self {
            return Self{};
        }

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            _ = allocator;
            _ = self;
        }

        pub fn forward(
            self: *@This(),
            inputs: []const f64,
            allocator: std.mem.Allocator,
        ) ![]f64 {
            self.inputs = inputs;

            var outputs = try allocator.alloc(f64, inputs.len);
            for (outputs, 0..) |_, index| {
                outputs[index] = activation_function.activate(-inputs[index]);
            }

            return outputs;
        }

        pub fn backward(
            self: *@This(),
            output_gradient: []const f64,
            allocator: std.mem.Allocator,
        ) ![]f64 {
            _ = allocator;
            // TODO

            for (self.inputs, 0..) |input, index| {
                _ = index;
                activation_function.derivative(input);
            }

            return output_gradient;
        }

        pub fn applyCostGradients(self: *Self, learn_rate: f64, options: ApplyCostGradientsOptions) void {
            // There are no parameters we need to update in an activation layer so this
            // is just a no-op.
            _ = self;
            _ = learn_rate;
            _ = options;
        }

        /// Helper to create a generic `Layer` that we can use in a `NerualNetwork`
        pub fn layer(self: *@This()) Layer {
            return Layer.init(self);
        }
    };
}
