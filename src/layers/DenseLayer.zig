//! "Dense" just means that every input is connected to every output. This is a "normal"
//! neural network layer. After each `DenseLayer`, the idiomatic thing to do is to add
//! an `ActivationLayer` to introduce non-linearity (can curve around the data to
//! classify things accurately).
//!
//! (inherits from `Layer`)
const std = @import("std");
const log = std.log.scoped(.zig_neural_networks);

const Layer = @import("Layer.zig");
const ActivationFunction = @import("../activation_functions.zig").ActivationFunction;

const InitializeWeightsAndBiasesOptions = struct {
    // It's nicer to have a fixed seed so we can reproduce the same results.
    random_seed: u64 = 123,
    // XXX: Even though we have this complexity to choose based on the activation
    // function, by default, we just assume that all of the layers should just use
    // Xavier initialization to avoid having to make sure that activation function
    // passed in to `init(...)` and the maintenance burden that it matches the
    // `ActivationLayer` after this layer.
    activation_function: ActivationFunction = ActivationFunction{ .sigmoid = .{} },
};

// pub const DenseLayer = struct {
const Self = @This();

pub const Parameters = struct {
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
};

parameters: Parameters,

/// Store the cost gradients for each weight and bias. These are used to update
/// the weights and biases after each training batch.
///
/// The partial derivative of the cost function with respect to the weight of the
/// current connection (dC/dw).
///
/// Size: num_output_nodes * num_input_nodes
cost_gradient_weights: []f64,
/// The partial derivative of the cost function with respect to the bias of the
/// current node (dC/db).
///
/// Size: num_output_nodes
cost_gradient_biases: []f64,

/// Used for adding momentum to gradient descent. Stores the change in weight/bias
/// from the previous learning iteration.
///
/// Size: num_output_nodes * num_input_nodes
weight_velocities: []f64,
/// Size: num_output_nodes
bias_velocities: []f64,

/// Store any inputs we get during the forward pass so we can use them during the
/// backward pass.
inputs: []const f64 = undefined,

pub fn init(
    num_input_nodes: usize,
    num_output_nodes: usize,
    allocator: std.mem.Allocator,
) !Self {
    // Initialize the weights
    const weights: []f64 = try allocator.alloc(f64, num_input_nodes * num_output_nodes);
    const biases: []f64 = try allocator.alloc(f64, num_output_nodes);
    // We're calling this with the defaults but feel free to call
    // `layer.initializeWeightsAndBiases()` again after the layer is created.
    Self._initializeWeightsAndBiases(weights, biases, .{});

    // Create the cost gradients and initialize the values to 0
    const cost_gradient_weights: []f64 = try allocator.alloc(f64, num_input_nodes * num_output_nodes);
    @memset(cost_gradient_weights, 0);
    const cost_gradient_biases: []f64 = try allocator.alloc(f64, num_output_nodes);
    @memset(cost_gradient_biases, 0);

    // Create the velocities and initialize the values to 0
    const weight_velocities: []f64 = try allocator.alloc(f64, num_input_nodes * num_output_nodes);
    @memset(weight_velocities, 0);
    const bias_velocities: []f64 = try allocator.alloc(f64, num_output_nodes);
    @memset(bias_velocities, 0);

    return Self{
        .parameters = .{
            .num_input_nodes = num_input_nodes,
            .num_output_nodes = num_output_nodes,
            .weights = weights,
            .biases = biases,
        },
        .cost_gradient_weights = cost_gradient_weights,
        .cost_gradient_biases = cost_gradient_biases,
        .weight_velocities = weight_velocities,
        .bias_velocities = bias_velocities,
    };
}

pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
    allocator.free(self.parameters.weights);
    allocator.free(self.parameters.biases);
    allocator.free(self.cost_gradient_weights);
    allocator.free(self.cost_gradient_biases);
    allocator.free(self.weight_velocities);
    allocator.free(self.bias_velocities);

    // This isn't strictly necessary but it marks the memory as dirty (010101...) in
    // safe modes (https://zig.news/kristoff/what-s-undefined-in-zig-9h)
    self.* = undefined;
}

/// Initialize the weights and biases for this layer. Weight initialization depends
/// on the activation function given.
pub fn initializeWeightsAndBiases(
    self: Self,
    options: InitializeWeightsAndBiasesOptions,
) void {
    Self._initializeWeightsAndBiases(self.parameters.weights, self.parameters.biases, options);
}

/// Internal implementation of `initializeWeightsAndBiases()` so we can use it in
/// the `init()` function where the layer isn't created yet.
fn _initializeWeightsAndBiases(
    weights: []f64,
    biases: []f64,
    options: InitializeWeightsAndBiasesOptions,
) void {
    var prng = std.rand.DefaultPrng.init(options.random_seed);
    const random_instance = prng.random();

    // Cheeky math so we can find the number of input nodes without referencing `self`.
    const num_input_nodes = weights.len / biases.len;

    // Initialize the weights of the network to random values
    for (weights) |*weight| {
        // Get a random value with a range `stddev = 1` centered around `mean = 0`.
        // When using a normal distribution like this, the odds are most likely that
        // your number will fall in the [-3, +3] range.
        //
        // > To use different parameters, use: floatNorm(...) * desiredStddev + desiredMean.
        const normal_random_value = random_instance.floatNorm(f64);
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
        switch (options.activation_function) {
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

/// Helper to access the weight for a specific connection since
/// the weights are stored in a flat array.
pub fn getWeight(self: *Self, node_index: usize, node_in_index: usize) f64 {
    const weight_index = self.getFlatWeightIndex(node_index, node_in_index);
    return self.parameters.weights[weight_index];
}

/// Helper to access the weight for a specific connection since
/// the weights are stored in a flat array.
pub fn getFlatWeightIndex(self: *Self, node_index: usize, node_in_index: usize) usize {
    return (node_index * self.parameters.num_input_nodes) + node_in_index;
}

/// Run the given `inputs` through the layer and return the outputs. To get the
/// `outputs`, each of the `inputs` are multiplied by the weight of their connection to
/// this layer and then the bias is added to the result.
///
/// y = x * w + b
///
/// - y is the output (also known as the weighted input or "z")
/// - x is the input
/// - w is the weight
/// - b is the bias
pub fn forward(
    self: *@This(),
    inputs: []const f64,
    allocator: std.mem.Allocator,
) ![]f64 {
    if (inputs.len != self.parameters.num_input_nodes) {
        log.err("DenseLayer.forward() was called with {d} inputs but we expect " ++
            "it to match the same num_input_nodes={d}", .{
            inputs.len,
            self.parameters.num_input_nodes,
        });

        return error.ExpectedInputLengthMismatch;
    }
    // Store the inputs so we can use them during the backward pass.
    self.inputs = inputs;

    // Calculate the weighted input sums for each node in this layer: w * x + b
    var outputs = try allocator.alloc(f64, self.parameters.num_output_nodes);
    for (0..self.parameters.num_output_nodes) |node_index| {
        // Calculate the weighted input for this node
        var weighted_input_sum: f64 = self.parameters.biases[node_index];
        for (0..self.parameters.num_input_nodes) |node_in_index| {
            weighted_input_sum += inputs[node_in_index] * self.getWeight(
                node_index,
                node_in_index,
            );
        }
        outputs[node_index] = weighted_input_sum;
    }

    return outputs;
}

/// Given the the derivative of the cost/loss/error with respect to the *outputs* (dC/dy),
/// returns the derivative of the cost/loss/error with respect to the *inputs* (dC/dx).
/// Also responsible for updating the cost gradients for the weights and biases.
///
/// (updateCostGradients and backward)
pub fn backward(
    self: *Self,
    /// The partial derivative of the cost/loss/error with respect to the *outputs*
    /// (dC/dy).
    ///
    /// Size: num_output_nodes
    output_gradient: []const f64,
    allocator: std.mem.Allocator,
) ![]f64 {
    if (output_gradient.len != self.parameters.num_output_nodes) {
        log.err("DenseLayer.backward() was called with a output_gradient of length {d} " ++
            "but we expect it to match the same num_output_nodes={d}", .{
            output_gradient.len,
            self.parameters.num_output_nodes,
        });

        return error.OutputGradientLengthMismatch;
    }

    // `input_gradient` stores the partial derivative of the cost (C) with respect
    // to the inputs (x) -> (dC/dx).
    // dC/dx = dC/dy * dy/dx
    // (derived via the chain rule)
    const input_gradient = try allocator.alloc(f64, self.parameters.num_input_nodes);

    // Update the cost gradients for the weights and biases.
    //
    // Given the standard equation in the forward pass:
    // y = x * w + b
    // where:
    // - y is the output (also known as the weighted input or "z")
    // - x is the input
    // - w is the weight
    // - b is the bias
    for (0..self.parameters.num_output_nodes) |node_index| {
        for (0..self.parameters.num_input_nodes) |node_in_index| {
            // Calculate the cost gradient for the weights (dC/dw)
            // ==========================================================

            // The partial derivative of the output (y) with respect to the weight
            // (w) -> (dy/dw). Given y = x * w + b, to find dy/dw, if we nudge the
            // weight (w), the output will change by the input (x).
            //
            // dy/dw = x
            const derivative_output_wrt_weight = self.inputs[node_in_index];
            // The partial derivative of cost (C) with respect to the weight (w) of
            // the current connection -> (dC/dw).
            // dC/dw = dy/dw * dC/dy
            // (derived via the chain rule)
            const derivative_cost_wrt_weight = derivative_output_wrt_weight * output_gradient[node_index];
            // The cost_gradient_weights array stores these partial derivatives for each weight.
            // Note: The derivative is being added to the array here because ultimately we want
            // to calculuate the average gradient across all the data in the training batch
            const flat_weight_index = self.getFlatWeightIndex(
                node_index,
                node_in_index,
            );
            self.cost_gradient_weights[flat_weight_index] += derivative_cost_wrt_weight;

            // Calculate the input gradient (dC/dx)
            // ==========================================================

            // The partial derivative of the output (y) with respect to the input
            // (x) -> (dy/dx). Given y = x * w + b, to find dy/dx, if we nudge the
            // input (x), the output will change by the weight (w).
            //
            // dy/dx = w
            const derivative_output_wrt_input = self.getWeight(
                node_index,
                node_in_index,
            );
            // dC/dx = dy/dx * dC/dy
            //
            // Essentialy, this is just a dot product between `output_gradient` and
            // the transposed weights matrix
            input_gradient[node_in_index] += derivative_output_wrt_input * output_gradient[node_index];
        }

        // Calculate the cost gradient for the biases (dC/db)
        // ==========================================================

        // The partial derivative of output (y) with respect to the bias (b) of
        // the current node -> (dy/db). Given y = x * w + b, to find dy/db, if we nudge
        // the bias (b), the output (y) will change by the same amount so dy/db = 1.
        const derivative_output_wrt_bias = 1;

        // The partial derivative of cost with respect to bias of the current node -> (dC/db).
        // dC/db = dy/db * dC/dy
        // (derived via the chain rule)
        const derivative_cost_wrt_bias = derivative_output_wrt_bias * output_gradient[node_index];
        self.cost_gradient_biases[node_index] += derivative_cost_wrt_bias;
    }

    return input_gradient;
}

/// Update the weights and biases based on the cost gradients (gradient descent).
/// Also should reset the cost gradients back to zero.
///
/// (also part of backward)
pub fn applyCostGradients(self: *Self, learn_rate: f64, options: Layer.ApplyCostGradientsOptions) void {
    // TODO: Implement weight decay (also known as or similar to "L2 regularization"
    // or "ridge regression") for purported effects that it "reduces overfitting"
    for (self.parameters.weights, 0..) |*weight, weight_index| {
        const velocity = (learn_rate * self.cost_gradient_weights[weight_index]) +
            (options.momentum * self.weight_velocities[weight_index]);
        // Store the velocity for use in the next iteration
        self.weight_velocities[weight_index] = velocity;

        // Update the weight
        weight.* -= velocity;
        // Reset the gradient back to zero now that we've applied it
        self.cost_gradient_weights[weight_index] = 0;
    }

    for (self.parameters.biases, 0..) |*bias, bias_index| {
        const velocity = (learn_rate * self.cost_gradient_biases[bias_index]) +
            (options.momentum * self.bias_velocities[bias_index]);
        // Store the velocity for use in the next iteration
        self.bias_velocities[bias_index] = velocity;

        // Update the bias
        bias.* -= velocity;
        // Reset the gradient back to zero now that we've applied it
        self.cost_gradient_biases[bias_index] = 0;
    }
}

/// Helper to create a generic `Layer` that we can use in a `NeuralNetwork`
pub fn layer(self: *@This()) Layer {
    return Layer.init(self);
}

/// Serialize the layer to JSON (using the `std.json` library).
pub fn jsonStringify(self: @This(), jws: anytype) !void {
    // What we output here, aligns with `Layer.SerializedLayer`. It's easier to use an
    // anonymous struct here instead of the `Layer.SerializedLayer` type because we know
    // the concrete type of the parameters here vs the generic `std.json.Value` from
    // `Layer.SerializedLayer`. Plus it's just more boilerplate for us to get
    // `self.parameters` into `std.json.Value` if we went that route.
    try jws.write(.{
        .serialized_type_name = @typeName(Self),
        .parameters = self.parameters,
    });
}

/// Deserialize the layer from JSON (using the `std.json` library).
pub fn jsonParse(allocator: std.mem.Allocator, source: anytype, options: std.json.ParseOptions) !@This() {
    const json_value = try std.json.parseFromTokenSourceLeaky(std.json.Value, allocator, source, options);
    return try jsonParseFromValue(allocator, json_value, options);
}

/// Deserialize the layer from a parsed JSON value. (using the `std.json` library).
pub fn jsonParseFromValue(allocator: std.mem.Allocator, source: std.json.Value, options: std.json.ParseOptions) !@This() {
    const parsed_parameters = try std.json.parseFromValue(
        Parameters,
        allocator,
        source,
        options,
    );
    defer parsed_parameters.deinit();
    const parameters = parsed_parameters.value;

    const dense_layer = try init(
        parameters.num_input_nodes,
        parameters.num_output_nodes,
        allocator,
    );
    @memcpy(dense_layer.parameters.weights, parameters.weights);
    @memcpy(dense_layer.parameters.biases, parameters.biases);

    return dense_layer;
}
