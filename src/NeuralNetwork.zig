const std = @import("std");
const log = std.log.scoped(.zig_neural_networks);
const tracy = @import("./tracy.zig");

const DataPoint = @import("./data_point.zig").DataPoint;
const argmax = @import("./data_point.zig").argmax;
const argmaxOneHotEncodedValue = @import("./data_point.zig").argmaxOneHotEncodedValue;
const Layer = @import("./layers/Layer.zig");
const DenseLayer = @import("./layers/DenseLayer.zig");
const ActivationLayer = @import("./layers/ActivationLayer.zig");
const ActivationFunction = @import("./activation_functions.zig").ActivationFunction;
const CostFunction = @import("./cost_functions.zig").CostFunction;

// pub const NeuralNetwork = struct {
const Self = @This();

layers: []Layer,
cost_function: CostFunction,

/// Keep track of any layers we specifically create in the NeuralNetwork from
/// functions like `initFromLayerSizes()` so we can free them when we `deinit`.
layers_to_free: struct {
    layers: ?[]Layer = null,
    dense_layers: ?[]DenseLayer = null,
    activation_layers: ?[]ActivationLayer = null,
},

/// Initializes a neural network from a list of layers. You should probably prefer
/// using `NeuralNetwork.initFromLayerSizes(...)` if you're just using the same
/// activation function for all layers except the output layer because it's a bit
/// more ergonomic and convenient.
///
/// Example usage:
/// ```
/// var dense_layer1 = try neural_networks.DenseLayer.init(2, 3, allocator);
/// var activation_layer1 = try neural_networks.ActivationLayer.init(neural_networks.ActivationFunction{ .sigmoid = .{} });
/// var dense_layer2 = try neural_networks.DenseLayer.init(3, 3, allocator);
/// var activation_layer2 = try neural_networks.ActivationLayer.init(neural_networks.ActivationFunction{ .elu = .{} });
/// var dense_layer3 = try neural_networks.DenseLayer.init(3, 2, allocator);
/// var activation_layer3 = try neural_networks.ActivationLayer.init(neural_networks.ActivationFunction{ .soft_max = .{} });
///
/// var layers = [_]neural_networks.Layer{
///     dense_layer1.layer(),
///     activation_layer1.layer(),
///     dense_layer2.layer(),
///     activation_layer2.layer(),
///     dense_layer3.layer(),
///     activation_layer3.layer(),
/// };
/// defer for (&layers) |*layer| {
///     layer.deinit(allocator);
/// };
//
/// var neural_network = try neural_networks.NeuralNetwork.initFromLayers(
///     &layers,
///     neural_networks.CostFunction{ .squared_error = .{} },
/// );
/// defer neural_network.deinit(allocator);
/// ```
pub fn initFromLayers(
    layers: []Layer,
    cost_function: CostFunction,
) !Self {
    const trace = tracy.trace(@src());
    defer trace.end();
    return Self{
        .layers = layers,
        .cost_function = cost_function,
        // We don't need to free any layers because it's the callers
        // responsibility to free them since they created them.
        .layers_to_free = .{},
    };
}

/// Convenience function for initializing a neural network from a list of layer sizes.
pub fn initFromLayerSizes(
    layer_sizes: []const u32,
    activation_function: ActivationFunction,
    output_layer_activation_function: ActivationFunction,
    cost_function: CostFunction,
    allocator: std.mem.Allocator,
) !Self {
    const trace = tracy.trace(@src());
    defer trace.end();
    const number_of_dense_layers = layer_sizes.len - 1;
    const output_dense_layer_index = number_of_dense_layers - 1;

    // We need to keep track of the specific layer types so they can live past
    // this stack context and so we can free them later on `deinit`.
    const dense_layers = try allocator.alloc(DenseLayer, number_of_dense_layers);
    const activation_layers = try allocator.alloc(ActivationLayer, number_of_dense_layers);

    // Create the list of layers in the network. Since we treat activation functions
    // as their own layer we create `DenseLayer` followed by `ActivationLayer`.
    var layers = try allocator.alloc(Layer, 2 * (number_of_dense_layers));
    for (dense_layers, activation_layers, 0..) |*dense_layer, *activation_layer, dense_layer_index| {
        const layer_activation_function = if (dense_layer_index == output_dense_layer_index)
            output_layer_activation_function
        else
            activation_function;

        // Create a dense layer.
        //
        // We need to create these on the heap otherwise they would just
        // disappear after we exit the stack and cause undefined behavior.
        dense_layer.* = try DenseLayer.init(
            layer_sizes[dense_layer_index],
            layer_sizes[dense_layer_index + 1],
            allocator,
        );
        // Initialize the weights specifically according to the activation function.
        // (the default initialization is probably fine but this is more fine-tuned)
        dense_layer.initializeWeightsAndBiases(.{ .activation_function = layer_activation_function });

        // We put an activation layer after every dense layer.
        activation_layer.* = try ActivationLayer.init(layer_activation_function);

        // Keep track of the generic layers
        const layer_index = 2 * dense_layer_index;
        layers[layer_index] = dense_layer.layer();
        layers[layer_index + 1] = activation_layer.layer();
    }

    return Self{
        .layers = layers,
        .cost_function = cost_function,
        .layers_to_free = .{
            .layers = layers,
            .dense_layers = dense_layers,
            .activation_layers = activation_layers,
        },
    };
}

pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
    const trace = tracy.trace(@src());
    defer trace.end();
    if (self.layers_to_free.layers) |layers| {
        for (layers) |*layer| {
            layer.deinit(allocator);
        }
        allocator.free(layers);
    }

    if (self.layers_to_free.dense_layers) |dense_layers| {
        allocator.free(dense_layers);
    }

    if (self.layers_to_free.activation_layers) |activation_layers| {
        allocator.free(activation_layers);
    }

    // This isn't strictly necessary but it marks the memory as dirty (010101...) in
    // safe modes (https://zig.news/kristoff/what-s-undefined-in-zig-9h)
    self.* = undefined;
}

/// Run the input values through the network to calculate the output values
/// (predict)
pub fn calculateOutputs(
    self: *Self,
    inputs: []const f64,
    allocator: std.mem.Allocator,
) ![]const f64 {
    const trace = tracy.trace(@src());
    defer trace.end();
    var inputs_to_next_layer = inputs;
    for (self.layers, 0..) |*layer, layer_index| {
        // Free the outputs from the last iteration at the end of the
        // block after we're done using it in the next layer.
        const inputs_to_free = inputs_to_next_layer;
        // Avoid freeing the initial `inputs` that someone passed in to this function.
        defer if (layer_index > 0) {
            allocator.free(inputs_to_free);
        };

        inputs_to_next_layer = try layer.forward(inputs_to_next_layer, allocator);
    }

    // We also avoid freeing the output of the last layer because we end up
    // returning it here.
    return inputs_to_next_layer;
}

/// Convience helper to calculate the accuracy of the network against a set of
/// testing data points.
///
/// Assumes one-hot encoded expected outputs for the data points.
pub fn getAccuracyAgainstTestingDataPoints(
    self: *Self,
    testing_data_points: []const DataPoint,
    allocator: std.mem.Allocator,
) !f64 {
    const trace = tracy.trace(@src());
    defer trace.end();
    var correct_count: f64 = 0;
    for (testing_data_points) |*testing_data_point| {
        const outputs = try self.calculateOutputs(testing_data_point.inputs, allocator);
        defer allocator.free(outputs);
        // argmax
        const max_output_index = argmax(outputs);

        // Assume one-hot encoded expected outputs
        const max_expected_outputs_index = try argmaxOneHotEncodedValue(testing_data_point.expected_outputs);

        if (max_output_index == max_expected_outputs_index) {
            correct_count += 1;
        }
    }

    return correct_count / @as(f64, @floatFromInt(testing_data_points.len));
}

/// Calculate the total cost of the network for a single data point
pub fn cost_individual(
    self: *Self,
    data_point: *const DataPoint,
    allocator: std.mem.Allocator,
) !f64 {
    const trace = tracy.trace(@src());
    defer trace.end();
    const outputs = try self.calculateOutputs(data_point.inputs, allocator);
    defer allocator.free(outputs);

    return self.cost_function.vector_cost(outputs, data_point.expected_outputs);
}

/// Calculate the total cost of the network for a batch of data points
pub fn cost_many(
    self: *Self,
    data_points: []const DataPoint,
    allocator: std.mem.Allocator,
) !f64 {
    const trace = tracy.trace(@src());
    defer trace.end();
    var total_cost: f64 = 0.0;
    for (data_points) |*data_point| {
        const cost_of_data_point = try self.cost_individual(data_point, allocator);
        total_cost += cost_of_data_point;
    }
    return total_cost;
}

/// Calculate the average cost of the network for a batch of data points
pub fn cost_average(
    self: *Self,
    data_points: []const DataPoint,
    allocator: std.mem.Allocator,
) !f64 {
    const trace = tracy.trace(@src());
    defer trace.end();
    const total_cost = try self.many_cost(data_points, allocator);
    return total_cost / @as(f64, @floatFromInt(data_points.len));
}

/// Run a single iteration of gradient descent.
/// We use gradient descent to minimize the cost function.
/// (train)
pub fn learn(
    self: *Self,
    training_data_batch: []const DataPoint,
    /// This configures how big of a step we make down the cost gradient landscape. This
    /// is usually a small value like `0.1` or `0.05` because large values can cause us
    /// to skip past the valley/hole where the minimum value of the cost is.
    learn_rate: f64,
    /// The momentum to apply to gradient descent. This is a value between 0 and 1
    /// and often has a value close to 1.0, such as 0.8, 0.9, or 0.99. A momentum of
    /// 0.0 is the same as gradient descent without momentum.
    //
    /// Momentum is used to help the gradient descent algorithm keep the learning
    /// process going in the right direction between different batches. It does this
    /// by adding a fraction of the previous weight change to the current weight
    /// change. Essentially, if it was moving before, it will keep moving in the same
    /// direction. It's most useful in situations where the cost surface has lots of
    /// curvature (changes a lot) ("highly non-spherical") or when the cost surface
    /// "flat or nearly flat, e.g. zero gradient. The momentum allows the search to
    /// progress in the same direction as before the flat spot and helpfully cross
    /// the flat region."
    /// (https://machinelearningmastery.com/gradient-descent-with-momentum-from-scratch/)
    //
    /// > The momentum algorithm accumulates an exponentially decaying moving average
    /// > of past gradients and continues to move in their direction.
    /// >
    /// > -- *Deep Learning* book page 296 (Ian Goodfellow)
    momentum: f64,
    allocator: std.mem.Allocator,
) !void {
    const trace = tracy.trace(@src());
    defer trace.end();
    // Use the backpropagation algorithm to calculate the gradient of the cost function
    // (with respect to the network's weights and biases). This is done for each data point,
    // and the gradients are added together.
    try self._updateCostGradients(training_data_batch, allocator);

    // Gradient descent step: update all weights and biases in the network
    try self._applyCostGradients(learn_rate, momentum, training_data_batch.len);
}

/// Gradient descent step: update all weights and biases in the network
///
/// This method is exposed publicly so that we can use it in tests for gradient
/// checking.
pub fn _applyCostGradients(self: *Self, learn_rate: f64, momentum: f64, training_batch_length: usize) !void {
    const trace = tracy.trace(@src());
    defer trace.end();
    for (self.layers) |*layer| {
        layer.applyCostGradients(
            // Because we summed the gradients from all of the training data points,
            // we need to average out all of gradients that we added together. Since
            // we end up multiplying the gradient values by the learn_rate, we can
            // just divide it by the number of training data points to get the
            // average gradient.
            learn_rate / @as(f64, @floatFromInt(training_batch_length)),
            .{
                .momentum = momentum,
            },
        );
    }
}

/// TODO: description
///
/// This method is exposed publicly so that we can use it in tests for gradient checking
/// but isn't meant for other public consumption.
pub fn _updateCostGradients(
    self: *Self,
    training_data_batch: []const DataPoint,
    allocator: std.mem.Allocator,
) !void {
    const trace = tracy.trace(@src());
    defer trace.end();
    // Use the backpropagation algorithm to calculate the gradient of the cost function
    // (with respect to the network's weights and biases). This is done for each data point,
    // and the gradients are added together.
    for (training_data_batch) |*data_point| {
        // Feed the data through the network to calculate the outputs. This also
        // allows the layers to save the inputs for use in the
        // `updateCostGradients`/backward step.
        //
        // This is similar to `calculateOuputs(...)` but we don't use it because the
        // layers need their inputs to stick around until after we're done with the
        // backward step.
        var inputs_to_next_layer = data_point.inputs;
        const layer_outputs_to_free_list = try allocator.alloc([]const f64, self.layers.len);
        // After we're done with the backward step we can free the layer outputs
        defer {
            for (layer_outputs_to_free_list) |layer_outputs_to_free| {
                allocator.free(layer_outputs_to_free);
            }
            allocator.free(layer_outputs_to_free_list);
        }
        for (self.layers, 0..) |*layer, layer_index| {
            const layer_outputs = try layer.forward(inputs_to_next_layer, allocator);
            inputs_to_next_layer = layer_outputs;
            layer_outputs_to_free_list[layer_index] = layer_outputs;
        }
        const outputs = inputs_to_next_layer;

        // ---- Backpropagation ----
        // Find the partial derivative of the cost/loss function with respect to the output
        // of the network -> (dC/dy)
        const cost_gradient = try allocator.alloc(f64, outputs.len);
        // (we free `cost_gradient` down below as part of the `output_gradient_for_next_layer` loop)
        for (
            cost_gradient,
            outputs,
            data_point.expected_outputs,
        ) |*cost_grad_element, output, expected_output| {
            cost_grad_element.* += self.cost_function.individual_derivative(
                output,
                expected_output,
            );
        }

        // Backpropagate the cost/loss gradient through the layers
        var output_gradient_for_next_layer = cost_gradient;
        var backward_layer_index: usize = self.layers.len - 1;
        while (backward_layer_index < self.layers.len) : (backward_layer_index -%= 1) {
            const layer = self.layers[backward_layer_index];

            // Free the output gradient from the last iteration at the end of the
            // block after we're done using it in the next layer.
            const output_gradient_to_free = output_gradient_for_next_layer;
            defer allocator.free(output_gradient_to_free);

            const input_gradient = try layer.backward(
                output_gradient_for_next_layer,
                allocator,
            );
            // Since the layers are chained together, the derivative of the cost
            // function with respect to the *input* of this layer is the same as the
            // derivative of the cost function with respect to the *output* of the
            // previous layer.
            output_gradient_for_next_layer = input_gradient;
        }
        // Free the last iteration of the loop
        defer allocator.free(output_gradient_for_next_layer);
    }
}

pub const SerializedNeuralNetwork = struct {
    timestamp: i64,
    layers: []Layer,
    cost_function: CostFunction,
};

/// Serialize the layer to JSON (using the `std.json` library).
pub fn jsonStringify(self: @This(), jws: anytype) !void {
    try jws.write(SerializedNeuralNetwork{
        .timestamp = std.time.timestamp(),
        .layers = self.layers,
        .cost_function = self.cost_function,
    });
}

/// Deserialize the layer from JSON (using the `std.json` library).
pub fn jsonParse(allocator: std.mem.Allocator, source: anytype, options: std.json.ParseOptions) !@This() {
    const json_value = try std.json.parseFromTokenSourceLeaky(std.json.Value, allocator, source, options);
    return try jsonParseFromValue(allocator, json_value, options);
}

/// Deserialize the layer from a parsed JSON value. (using the `std.json` library).
pub fn jsonParseFromValue(allocator: std.mem.Allocator, source: std.json.Value, options: std.json.ParseOptions) !@This() {
    const parsed_serialized_neural_network = try std.json.parseFromValue(
        SerializedNeuralNetwork,
        allocator,
        source,
        options,
    );
    defer parsed_serialized_neural_network.deinit();
    const serialized_neural_network = parsed_serialized_neural_network.value;

    return .{
        .layers = serialized_neural_network.layers,
        .cost_function = serialized_neural_network.cost_function,
        .layers_to_free = .{
            .layers = serialized_neural_network.layers,
        },
    };
}

test {
    _ = @import("./tests/neural_network_tests.zig");
    _ = @import("./tests/neural_network_serialization_tests.zig");
}
