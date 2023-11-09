const std = @import("std");
const log = std.log.scoped(.zig_neural_networks);
const shuffle = @import("zshuffle").shuffle;

pub const LabelType = union(enum) {
    int: i64,
    float: f64,
    string: []const u8,

    const Tag = std.meta.Tag(LabelType);
    pub fn tag(u: LabelType) Tag {
        return @as(Tag, u);
    }
};

pub const DataPoint = struct {
    const Self = @This();

    labels: []const LabelType,
    inputs: []const f64,
    expected_outputs: []const f64,

    pub fn init(inputs: []const f64, label: LabelType, labels: []const LabelType) Self {
        return .{
            .inputs = inputs,
            .expected_outputs = oneHotEncodeLabel(label, labels),
            .labels = labels,
        };
    }

    fn oneHotEncodeLabel(label: LabelType, labels: []const LabelType) []f64 {
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

    // This is just complicated logic to handle numbers or strings as labels
    pub fn checkLabelsEqual(a: LabelType, b: LabelType) bool {
        if (a.tag() != b.tag()) {
            return false;
        }

        const is_label_matching = switch (a) {
            .int, .float => a == b,
            .string => std.mem.eql(u8, a, b),
        };

        return is_label_matching;
    }
};

// It's nicer to have a fixed seed so we can reproduce the same results.
const seed = 123;
var prng = std.rand.DefaultPrng.init(seed);
const default_random_instance = prng.random();

pub fn shuffleData(data: anytype, allocator: std.mem.Allocator, options: struct {
    random_instance: std.rand.Random = default_random_instance,
}) ![]@TypeOf(data[0]) {
    return try shuffle(options.random_instance, data, .{
        .allocator = allocator,
    });
}
