const std = @import("std");

const Layer = @import("./layers/layer.zig").Layer;
const ActivationFunction = @import("./activation_functions.zig").ActivationFunction;
const LossFunction = @import("./loss_functions.zig").LossFunction;

pub const NeuralNetwork = struct {
    const Self = @This();

    /// ex.
    /// ```
    /// var dense_layer1 = neural_network.DenseLayer.init(2, 3, allocator);
    /// defer dense_layer1.deinit();
    /// var activation_layer1 = neural_network.ActivationLayer(neural_network.ActivationFunction{ .elu = {} }).init();
    /// var dense_layer2 = neural_network.DenseLayer.init(3, 2);
    /// defer dense_layer2.deinit();
    /// var activation_layer2 = neural_network.ActivationLayer(neural_network.ActivationFunction{ .soft_max = {} }).init();
    //
    /// var layers = [_]neural_network.Layer{
    ///     dense_layer1.layer(),
    ///     activation_layer1.layer(),
    ///     dense_layer2.layer(),
    ///     activation_layer2.layer(),
    /// };
    //
    /// neural_network.NeuralNetwork.init(
    ///     layers,
    ///     neural_network.LossFunction{ .squared_error = {} },
    ///     allocator,
    /// );
    /// ```
    pub fn initFromLayers(
        layers: []const Layer,
        loss_function: LossFunction,
        allocator: std.mem.Allocator,
    ) !Self {
        _ = allocator;
        _ = loss_function;
        _ = layers;
    }

    pub fn initFromLayerSizes(
        layer_sizes: []const u32,
        activation_function: ActivationFunction,
        output_layer_activation_function: ActivationFunction,
        loss_function: LossFunction,
        allocator: std.mem.Allocator,
    ) !Self {
        _ = allocator;
        _ = loss_function;
        _ = output_layer_activation_function;
        _ = activation_function;
        _ = layer_sizes;
    }
};
