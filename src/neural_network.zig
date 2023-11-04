const std = @import("std");

const Layer = @import("./layers/layer.zig").Layer;
const DenseLayer = @import("./layers/dense_layer.zig").DenseLayer;
const ActivationLayer = @import("./layers/activation_layer.zig").ActivationLayer;
const ActivationFunction = @import("./activation_functions.zig").ActivationFunction;
const CostFunction = @import("./cost_functions.zig").CostFunction;

pub fn NeuralNetwork(comptime DataPointType: type) type {
    return struct {
        const Self = @This();

        layers: []Layer,
        cost_function: CostFunction,

        /// Initializes a neural network from a list of layers. You should probably prefer
        /// using `NeuralNetwork.initFromLayerSizes(...)` if you're just using the same
        /// activation function for all layers except the output layer because it's a bit
        /// more ergonomic and convenient.
        ///
        /// Example usage:
        /// ```
        /// var dense_layer1 = neural_network.DenseLayer.init(2, 3, allocator);
        /// var activation_layer1 = neural_network.ActivationLayer(neural_network.ActivationFunction{ .elu = {} }).init();
        /// var dense_layer2 = neural_network.DenseLayer.init(3, 2);
        /// var activation_layer2 = neural_network.ActivationLayer(neural_network.ActivationFunction{ .soft_max = {} }).init();
        //
        /// var layers = [_]neural_network.Layer{
        ///     dense_layer1.layer(),
        ///     activation_layer1.layer(),
        ///     dense_layer2.layer(),
        ///     activation_layer2.layer(),
        /// };
        /// defer {
        ///     for (layers) |*layer| {
        ///         layer.deinit(allocator);
        ///     }
        /// }
        //
        /// neural_network.NeuralNetwork.initFromLayers(
        ///     layers,
        ///     neural_network.CostFunction{ .squared_error = {} },
        /// );
        /// ```
        pub fn initFromLayers(
            layers: []const Layer,
            cost_function: CostFunction,
        ) !Self {
            return Self{
                .layers = layers,
                .cost_function = cost_function,
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
            const number_of_dense_layers = layer_sizes.len - 1;
            const output_dense_layer_index = number_of_dense_layers - 1;

            // Create the list of layers in the network. Since we treat activation functions
            // as their own layer we create `DenseLayer` followed by `ActivationLayer`.
            var layers = try allocator.alloc(Layer, 2 * (number_of_dense_layers));
            for (0..number_of_dense_layers) |dense_layer_index| {
                const layer_activation_function = if (dense_layer_index == output_dense_layer_index)
                    output_layer_activation_function
                else
                    activation_function;

                // Create a dense layer.
                var dense_layer = try DenseLayer.init(
                    layer_sizes[dense_layer_index],
                    layer_sizes[dense_layer_index + 1],
                    allocator,
                );
                dense_layer.initializeWeightsAndBiases(.{ .activation_function = layer_activation_function });
                // We put an activation layer after every dense layer.
                var activation_layer = try ActivationLayer(layer_activation_function).init();

                const layer_index = 2 * dense_layer_index;
                layers[layer_index] = dense_layer.layer();
                layers[layer_index + 1] = activation_layer.layer();
            }

            return Self{
                .layers = layers,
                .cost_function = cost_function,
            };
        }

        pub fn deinitFromLayerSizes(self: *Self, allocator: std.mem.Allocator) void {
            for (self.layers) |*layer| {
                layer.deinit(allocator);
            }

            allocator.free(self.layers);
        }

        /// Run the input values through the network to calculate the output values
        /// (predict)
        pub fn calculateOutputs(
            self: *Self,
            inputs: []const f64,
            allocator: std.mem.Allocator,
        ) ![]const f64 {
            var inputs_to_next_layer = inputs;
            for (self.layers) |*layer| {
                inputs_to_next_layer = try layer.forward(inputs_to_next_layer, allocator);
            }

            return inputs_to_next_layer;
        }

        /// Layers need to keep track of the input in the forward direction so we can re-use
        /// it in the backward direction. This function frees the inputs of all the layers
        /// after we do our backward calculations.
        pub fn freeAfterCalculateOutputs(self: *Self, allocator: std.mem.Allocator) void {
            // We only need to free the inputs for the hidden layers because the output
            // of one layer is the input to the next layer. We do need to clean up the
            // output of the output layer though (see below).
            for (self.layers, 0..) |*layer, layer_index| {
                // Avoid freeing the initial `inputs` that someone passed in to this function.
                if (layer_index > 0) {
                    allocator.free(layer.inputs);
                }
            }
        }

        /// Run a single iteration of gradient descent.
        /// We use gradient descent to minimize the cost function.
        /// (train)
        pub fn learn(
            self: *Self,
            training_data_batch: []const DataPointType,
            learn_rate: f64,
            /// See the comment in `Layer.updateCostGradients()` for more info
            momentum: f64,
            allocator: std.mem.Allocator,
        ) !void {
            for (training_data_batch) |data_point| {
                try self.updateCostGradients(data_point, allocator);
            }

            // TODO: Gradient check

            // Gradient descent step: update all weights and biases in the network
            for (self.layers) |*layer| {
                layer.applyCostGradients(
                    // Because we summed the gradients from all of the training data points,
                    // we need to average out all of gradients that we added together. Since
                    // we end up multiplying the gradient values by the learn_rate, we can
                    // just divide it by the number of training data points to get the
                    // average gradient.
                    learn_rate / @as(f64, @floatFromInt(training_data_batch.len)),
                    momentum,
                );
            }
        }

        /// TODO: description
        fn updateCostGradients(
            self: *Self,
            data_point: *const DataPointType,
            allocator: std.mem.Allocator,
        ) !void {
            const outputs = try self.calculateOutputs(data_point.inputs, allocator);
            defer allocator.free(outputs);
            defer self.freeAfterCalculateOutputs(allocator);

            // Find the partial derivative of the loss function with respect to the output
            // of the network.
            const loss_gradient = allocator.alloc(f64, outputs.len);
            defer allocator.free(loss_gradient);
            for (
                loss_gradient,
                outputs,
                data_point.expected_outputs,
            ) |*loss_grad_element, output, expected_output| {
                loss_grad_element.* += try self.cost_function.individual_derivative(
                    output,
                    expected_output,
                );
            }

            // Backpropagate the loss gradient through the layers
            var output_gradient_for_next_layer = loss_gradient;
            var backward_layer_index: u32 = self.layers.len - 1;
            while (backward_layer_index < self.layers.len) : (backward_layer_index -%= 1) {
                const layer = self.layers[backward_layer_index];

                // Free the shareable_node_derivatives from the last iteration at the
                // end of the block after we're done using it in the next hidden layer.
                const output_gradient_to_free = output_gradient_for_next_layer;
                defer allocator.free(output_gradient_to_free);

                output_gradient_for_next_layer = try layer.updateCostGradients(
                    output_gradient_for_next_layer,
                    allocator,
                );
            }
            // Free the last iteration of the loop
            defer allocator.free(output_gradient_for_next_layer);
        }
    };
}
