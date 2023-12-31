const std = @import("std");
const log = std.log.scoped(.zig_neural_networks);
const tracy = @import("./tracy.zig");
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
pub fn shuffleData(
    /// `[]DataPoint` to shuffle. This function happens to work with practically with
    /// any type but we have it here to work with a slice of `DataPoint`.
    data: anytype,
    allocator: std.mem.Allocator,
    options: struct {
        random_instance: std.rand.Random = default_random_instance,
    },
) ![]@TypeOf(data[0]) {
    const trace = tracy.trace(@src());
    defer trace.end();
    return try shuffle(options.random_instance, data, .{
        .allocator = allocator,
    });
}

/// Find the index of the maximum value in the outputs array.
pub fn argmax(outputs: []const f64) usize {
    var max_output_index: usize = 0;
    for (outputs, 0..) |output, index| {
        if (output > outputs[max_output_index]) {
            max_output_index = index;
        }
    }

    return max_output_index;
}

/// Find the index of the first value that is 1.0 in the outputs array.
pub fn argmaxOneHotEncodedValue(one_hot_outputs: []const f64) !usize {
    var one_hot_index: usize = 0;
    for (one_hot_outputs, 0..) |output, index| {
        if (output == 1.0) {
            one_hot_index = index;
            break;
        } else if (output != 0.0) {
            log.err("Value is not one-hot encoded ({any}) " ++
                "but `argmaxOneHotEncodedValue(...)` assumes that they should be.", .{
                one_hot_outputs,
            });
            return error.ValueIsNotOneHotEncoded;
        }
    }

    return one_hot_index;
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
///     // FIXME: Once https://github.com/ziglang/zig/pull/18112 merges and we support a Zig
///     // version that includes it, we should use `getPtrConstAssertContains(...)` instead.
///     one_hot_iris_flower_label_map.getPtrConst(.virginica).?,
/// );
/// ```
pub fn convertLabelEnumToOneHotEncodedEnumMap(
    comptime EnumType: type,
) std.EnumMap(EnumType, [@typeInfo(EnumType).Enum.fields.len]f64) {
    const EnumTypeInfo = @typeInfo(EnumType);
    const num_enum_fields = EnumTypeInfo.Enum.fields.len;

    var label_to_one_hot_encoded_value_map = std.EnumMap(EnumType, [num_enum_fields]f64).initFull(
        std.mem.zeroes([num_enum_fields]f64),
    );

    inline for (EnumTypeInfo.Enum.fields, 0..) |field, field_index| {
        var one_hot = std.mem.zeroes([num_enum_fields]f64);
        one_hot[field_index] = 1.0;
        label_to_one_hot_encoded_value_map.put(@field(EnumType, field.name), one_hot);
    }

    return label_to_one_hot_encoded_value_map;
}

const IrisFlowerLabel = enum {
    virginica,
    versicolor,
    setosa,
};
const comptime_one_hot_iris_flower_label_map = convertLabelEnumToOneHotEncodedEnumMap(IrisFlowerLabel);

test "convertLabelEnumToOneHotEncodedEnumMap at comptime" {
    try std.testing.expectEqualSlices(
        f64,
        &[_]f64{ 1.0, 0.0, 0.0 },
        comptime_one_hot_iris_flower_label_map.getPtrConst(.virginica).?,
    );
    try std.testing.expectEqualSlices(
        f64,
        &[_]f64{ 0.0, 1.0, 0.0 },
        comptime_one_hot_iris_flower_label_map.getPtrConst(.versicolor).?,
    );
    try std.testing.expectEqualSlices(
        f64,
        &[_]f64{ 0.0, 0.0, 1.0 },
        comptime_one_hot_iris_flower_label_map.getPtrConst(.setosa).?,
    );
}

// Just sanity check that we're not returning pointers to stack memory.
test "convertLabelEnumToOneHotEncodedEnumMap(...) at runtime" {
    const runtime_one_hot_iris_flower_label_map = convertLabelEnumToOneHotEncodedEnumMap(IrisFlowerLabel);

    try std.testing.expectEqualSlices(
        f64,
        &[_]f64{ 1.0, 0.0, 0.0 },
        runtime_one_hot_iris_flower_label_map.getPtrConst(.virginica).?,
    );
    try std.testing.expectEqualSlices(
        f64,
        &[_]f64{ 0.0, 1.0, 0.0 },
        runtime_one_hot_iris_flower_label_map.getPtrConst(.versicolor).?,
    );
    try std.testing.expectEqualSlices(
        f64,
        &[_]f64{ 0.0, 0.0, 1.0 },
        runtime_one_hot_iris_flower_label_map.getPtrConst(.setosa).?,
    );
}
