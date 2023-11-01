const std = @import("std");
const zig_neural_network = @import("zig-neural-network");

pub const std_options = struct {
    pub const log_level = .debug;

    pub const log_scope_levels = &[_]std.log.ScopeLevel{
        .{ .scope = .zig_neural_network, .level = .debug },
    };
};

pub fn main() void {
    std.log.debug("adsf {d}", .{zig_neural_network.add(1, 2)});
}
