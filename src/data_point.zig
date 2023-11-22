const std = @import("std");
const log = std.log.scoped(.zig_neural_networks);
const shuffle = @import("zshuffle").shuffle;

pub const DataPoint = struct {
    const Self = @This();

    inputs: []const f64,
    expected_outputs: []const f64,

    pub fn init(inputs: []const f64, expected_outputs: []const f64) Self {
        return .{
            .inputs = inputs,
            .expected_outputs = expected_outputs,
        };
    }
};

// It's nicer to have a fixed seed so we can reproduce the same results.
const seed = 123;
var prng = std.rand.DefaultPrng.init(seed);
const default_random_instance = prng.random();

// This turns out just to be a small wrapper around zshuffle so people don't have to
// deal with the random instance boilerplate. (potentially, this is not a useful
// abstraction)
pub fn shuffleData(data: anytype, allocator: std.mem.Allocator, options: struct {
    random_instance: std.rand.Random = default_random_instance,
}) ![]@TypeOf(data[0]) {
    return try shuffle(options.random_instance, data, .{
        .allocator = allocator,
    });
}

/// Helper function to convert an label enum type (all of the possible labels in a neural
/// network) to a map from the enum label to a one-hot encoded value. The one-hot 1.0
/// value is at the index of the enum label in the enum definition.
///
/// Usage:
/// ```
/// const IrisFlowerLabel = enum {
///     virginica,
///     versicolor,
///     setosa,
/// };
/// const one_hot_iris_flower_label_map = convertLabelEnumToOneHotEncodedEnumMap(IrisFlowerLabel);
/// const example_data_point = DataPoint.init(
///     &[_]f64{ 7.2, 3.6, 6.1, 2.5 },
///     one_hot_iris_flower_label_map.getAssertContains(.virginica),
/// );
/// ```
pub fn convertLabelEnumToOneHotEncodedEnumMap(comptime EnumType: type) std.EnumMap(EnumType, []const f64) {
    var label_to_one_hot_encoded_value_map = std.EnumMap(EnumType, []const f64).initFull(&[_]f64{});

    const EnumTypeInfo = @typeInfo(EnumType);
    inline for (EnumTypeInfo.Enum.fields, 0..) |field, field_index| {
        var one_hot = std.mem.zeroes([EnumTypeInfo.Enum.fields.len]f64);
        one_hot[field_index] = 1.0;
        label_to_one_hot_encoded_value_map.put(@field(EnumType, field.name), &one_hot);
    }

    return label_to_one_hot_encoded_value_map;
}
