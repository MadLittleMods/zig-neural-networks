const std = @import("std");
const log = std.log.scoped(.zig_neural_network);

pub const NeuralNetwork = @import("./neural_network.zig").NeuralNetwork;

pub const Layer = @import("./layers/layer.zig").Layer;
pub const DenseLayer = @import("./layers/dense_layer.zig").DenseLayer;
pub const ActivationLayer = @import("./layers/activation_layer.zig").ActivationLayer;

pub const ActivationFunction = @import("./activation_functions.zig").ActivationFunction;
pub const LossFunction = @import("./loss_functions.zig").LossFunction;
