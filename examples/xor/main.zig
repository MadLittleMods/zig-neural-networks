const std = @import("std");
const zig_neural_network = @import("zig-neural-network");

pub const std_options = struct {
    pub const log_level = .debug;

    pub const log_scope_levels = &[_]std.log.ScopeLevel{
        .{ .scope = .zig_neural_network, .level = .debug },
    };
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    _ = allocator;
    defer {
        switch (gpa.deinit()) {
            .ok => {},
            .leak => std.log.err("GPA allocator: Memory leak detected", .{}),
        }
    }

    var layers = [_]zig_neural_network.Layer{
        (zig_neural_network.DenseLayer{}).layer(),
        (zig_neural_network.DenseLayer{}).layer(),
        (zig_neural_network.DenseLayer{}).layer(),
    };

    std.log.debug("layers {any}", .{layers});
}
