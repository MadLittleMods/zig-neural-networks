const std = @import("std");
const log = std.log.scoped(.zig_neural_network);

const Layer = @import("layer.zig").Layer;
const ActivationFunction = @import("../activation_functions.zig").ActivationFunction;

/// "Dense" just means that every input is connected to every output. This is a "normal"
/// neural network layer.
///
/// (inherits from `Layer`)
pub const DenseLayer = struct {
    const Self = @This();
    num_input_nodes: usize,
    num_output_nodes: usize,
    /// Weights for each incoming connection. Each node in this layer has a weighted
    /// connection to each node in the previous layer (num_input_nodes *
    /// num_output_nodes).
    ///
    /// The weights are stored in row-major order where each row is the incoming
    /// connection weights for a single node in this layer.
    ///
    /// Size: num_output_nodes * num_input_nodes
    weights: []f64,
    /// Bias for each node in the layer (num_output_nodes)
    ///
    /// Size: num_output_nodes
    biases: []f64,

    /// Store any inputs we get during the forward pass so we can use them during the
    /// backward pass.
    inputs: []f64 = undefined,

    pub fn init(
        num_input_nodes: usize,
        num_output_nodes: usize,
        allocator: std.mem.Allocator,
    ) !Self {
        // Initialize the weights
        var weights: []f64 = try allocator.alloc(f64, num_input_nodes * num_output_nodes);
        var biases: []f64 = try allocator.alloc(f64, num_output_nodes);
        Self.initializeWeightsAndBiases(
            weights,
            biases,
            num_input_nodes,
        );

        return Self{
            .num_input_nodes = num_input_nodes,
            .num_output_nodes = num_output_nodes,
            .weights = weights,
            .biases = biases,
        };
    }

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        allocator.free(self.weights);
        allocator.free(self.biases);
    }

    fn initializeWeightsAndBiases(
        weights: []f64,
        biases: []f64,
    ) void {
        // It's nicer to have a fixed seed so we can reproduce the same results.
        const seed = 123;
        var prng = std.rand.DefaultPrng.init(seed);

        // Cheeky math so we can find the number of input nodes without referencing `self`.
        const num_input_nodes = weights.len / biases.len;

        // XXX: Even though we have this complexity to choose based on the
        // activation function, we're assuming that all of the layers should just
        // use Xavier initialization to avoid having to make sure that activation
        // function passed in here matches the `ActivationLayer` after this layer.
        const activation_function = ActivationFunction{ .sigmoid = {} };

        // Initialize the weights of the network to random values
        for (weights) |*weight| {
            // Get a random value with a range `stddev = 1` centered around `mean = 0`.
            // When using a normal distribution like this, the odds are most likely that
            // your number will fall in the [-3, +3] range.
            //
            // > To use different parameters, use: floatNorm(...) * desiredStddev + desiredMean.
            const normal_random_value = prng.random().floatNorm(f64);
            // Now to choose a good weight initialization scheme. The "best" heuristic
            // often depends on the specific activiation function being used. We want to
            // avoid the vanishing/exploding gradient problem.
            //
            // Xavier initialization takes a set of random values sampled uniformly from
            // a range proportional to the size of the number of nodes in the previous
            // layer (fan-in). Specifically multiplying the normal random value by
            // `stddev = sqrt(1 / fan_in)`.
            //
            // "He initialization" is similar to Xavier initialization, but multiplies
            // the normal random value by `stddev = sqrt(2 / fan_in)`. This modification
            // is suggested when using the ReLU activation function to achieve a
            // "properly scaled uniform distribution for initialization".
            //
            // References:
            //  - https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/
            //  - https://prateekvishnu.medium.com/xavier-and-he-normal-he-et-al-initialization-8e3d7a087528
            const desired_mean = 0;
            switch (activation_function) {
                ActivationFunction.relu,
                ActivationFunction.leaky_relu,
                ActivationFunction.elu,
                => {
                    // He initialization
                    const desired_standard_deviation = @sqrt(2.0 /
                        @as(f64, @floatFromInt(num_input_nodes)));
                    weight.* = normal_random_value * desired_standard_deviation + desired_mean;
                },
                else => {
                    // Xavier initialization
                    const desired_standard_deviation = @sqrt(1.0 /
                        @as(f64, @floatFromInt(num_input_nodes)));
                    weight.* = normal_random_value * desired_standard_deviation + desired_mean;
                },
            }

            // Note: there are many different ways of trying to chose a good range for
            // the random weights, and these depend on facors such as the activation
            // function being used. and how the inputs to the network have been
            // scaled/normalized (ideally our input data should be scaled to the range
            // [0, 1]).
            //
            // For example, when using the sigmoid activation function, we don't want
            // the weighted inputs to be too large, as otherwise the slope of the
            // function will be very close to zero, resulting in the gradient descent
            // algorithm learning very slowly (or not at all).
        }

        for (biases) |*bias| {
            // Specifically for the ReLU activation function, the *Deep Learning* (Ian
            // Goodfellow) book suggests:
            // > it can be a good practice to set all elements of [the bias] to a small,
            // > positive value, such as 0.1. This makes it very likely that the rectified
            // > linear units will be initially active for most inputs in the training set
            // > and allow the derivatives to pass through.
            bias.* = 0.1;
        }
    }

    pub fn forward(
        self: *@This(),
        inputs: []const f64,
        allocator: std.mem.Allocator,
    ) ![]f64 {
        if (inputs.len != self.num_input_nodes) {
            log.err("DenseLayer.forward() was called with {d} inputs but we expect " ++
                "it to match the same num_input_nodes={d}", .{
                inputs.len,
                self.num_input_nodes,
            });

            return error.ExpectedOutputCountMismatch;
        }
        self.inputs = inputs;

        // Calculate the weighted input sums for each node in this layer: w * x + b
        var outputs = try allocator.alloc(f64, self.num_output_nodes);
        for (0..self.num_output_nodes) |node_index| {
            // Calculate the weighted input for this node
            var weighted_input_sum: f64 = self.biases[node_index];
            for (0..self.num_input_nodes) |node_in_index| {
                weighted_input_sum += inputs[node_in_index] * self.getWeight(
                    node_index,
                    node_in_index,
                );
            }
            outputs[node_index] = weighted_input_sum;
        }

        return outputs;
    }

    pub fn backward(
        self: *@This(),
        output_gradient: []const f64,
        learn_rate: f64,
        allocator: std.mem.Allocator,
    ) ![]f64 {
        _ = learn_rate;
        _ = allocator;
        _ = self;
        // TODO

        return output_gradient;
    }

    /// Helper to create a generic `Layer` that we can use in a `NerualNetwork`
    pub fn layer(self: *@This()) Layer {
        return Layer.init(self);
    }

    /// Helper to access the weight for a specific connection since
    /// the weights are stored in a flat array.
    pub fn getWeight(self: *Self, node_index: usize, node_in_index: usize) f64 {
        const weight_index = self.getFlatWeightIndex(node_index, node_in_index);
        return self.weights[weight_index];
    }

    /// Helper to access the weight for a specific connection since
    /// the weights are stored in a flat array.
    pub fn getFlatWeightIndex(self: *Self, node_index: usize, node_in_index: usize) usize {
        return (node_index * self.num_input_nodes) + node_in_index;
    }
};
