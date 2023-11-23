const std = @import("std");
const log = std.log.scoped(.zig_neural_networks);
const tracy = @import("./tracy.zig");
// const shuffle = @import("zshuffle").shuffle;

pub fn DataPoint(
    /// The type of the label. This can be an integer, float, or string (`[]const u8`).
    comptime InputLabelType: type,
    /// The possible labels for a data point.
    comptime labels: []const InputLabelType,
) type {
    return struct {
        const Self = @This();

        pub const LabelType = InputLabelType;
        pub const label_list = labels;

        inputs: []const f64,
        expected_outputs: [labels.len]f64,
        label: InputLabelType,

        pub fn init(inputs: []const f64, label: InputLabelType) Self {
            return .{
                .inputs = inputs,
                .expected_outputs = oneHotEncodeLabel(label),
                .label = label,
            };
        }

        fn oneHotEncodeLabel(label: InputLabelType) [labels.len]f64 {
            var one_hot = std.mem.zeroes([labels.len]f64);
            for (labels, 0..) |comparison_label, label_index| {
                const is_label_matching = checkLabelsEqual(comparison_label, label);
                if (is_label_matching) {
                    one_hot[label_index] = 1.0;
                } else {
                    one_hot[label_index] = 0.0;
                }
            }
            return one_hot;
        }

        pub fn oneHotIndexToLabel(one_hot_index: usize) InputLabelType {
            return labels[one_hot_index];
        }

        pub fn labelToOneHotIndex(label: InputLabelType) !usize {
            for (labels, 0..) |comparison_label, label_index| {
                const is_label_matching = checkLabelsEqual(comparison_label, label);
                if (is_label_matching) {
                    return label_index;
                }
            }

            switch (@typeInfo(InputLabelType)) {
                .Int, .Float => {
                    log.err("Unable to find label {d} in label list {any}", .{ label, labels });
                },
                .Pointer => |ptr_info| {
                    if (!ptr_info.is_const or ptr_info.size != .Slice or ptr_info.child != u8) {
                        @compileError("unsupported type");
                    }

                    // We found the label to be a string (`[]const u8`)
                    log.err("Unable to find label {s} in label list {any}", .{ label, labels });
                },
                else => @compileError("unsupported type"),
            }
            return error.LabelNotFound;
        }

        // This is just complicated logic to handle numbers or strings as labels
        pub fn checkLabelsEqual(a: InputLabelType, b: InputLabelType) bool {
            const is_label_matching = switch (@typeInfo(InputLabelType)) {
                .Int, .Float => a == b,
                .Pointer => |ptr_info| blk: {
                    if (!ptr_info.is_const or ptr_info.size != .Slice or ptr_info.child != u8) {
                        @compileError("unsupported type");
                    }

                    // Compare strings (`[]const u8`)
                    break :blk std.mem.eql(u8, a, b);
                },
                else => @compileError("unsupported type"),
            };

            return is_label_matching;
        }
    };
}

// It's nicer to have a fixed seed so we can reproduce the same results.
// const seed = 123;
// var prng = std.rand.DefaultPrng.init(seed);
// const default_random_instance = prng.random();

// TODO: Restore functionality once https://github.com/hmusgrave/zshuffle/pull/2 merges
// pub fn shuffleData(data: anytype, allocator: std.mem.Allocator, options: struct {
//     random_instance: std.rand.Random = default_random_instance,
// }) ![]@TypeOf(data[0]) {
//     const trace = tracy.trace(@src());
//     defer trace.end();
//     return try shuffle(options.random_instance, data, .{
//         .allocator = allocator,
//     });
// }
