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
