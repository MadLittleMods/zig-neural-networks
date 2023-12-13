const std = @import("std");
const log = std.log.scoped(.zig_neural_networks);

pub const NeuralNetwork = @import("./NeuralNetwork.zig");

pub const Layer = @import("./layers/Layer.zig");
pub const DenseLayer = @import("./layers/DenseLayer.zig");
pub const ActivationLayer = @import("./layers/ActivationLayer.zig");

pub const ActivationFunction = @import("./activation_functions.zig").ActivationFunction;
pub const CostFunction = @import("./cost_functions.zig").CostFunction;

pub const DataPoint = @import("./data_point.zig").DataPoint;
pub const shuffleData = @import("./data_point.zig").shuffleData;
pub const argmax = @import("./data_point.zig").argmax;
pub const argmaxOneHotEncodedValue = @import("./data_point.zig").argmaxOneHotEncodedValue;
pub const convertLabelEnumToOneHotEncodedEnumMap = @import("./data_point.zig").convertLabelEnumToOneHotEncodedEnumMap;

test {
    // https://ziglang.org/documentation/master/#Nested-Container-Tests
    @import("std").testing.refAllDecls(@This());
}
