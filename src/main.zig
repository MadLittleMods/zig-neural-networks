const std = @import("std");
const testing = std.testing;
const log = std.log.scoped(.zig_neural_network);

pub fn add(a: i32, b: i32) i32 {
    log.debug("add() called", .{});
    return a + b;
}

test "basic add functionality" {
    try testing.expect(add(3, 7) == 10);
}
