const std = @import("std");
const log = std.log.scoped(.zig_neural_network);

pub const Layer = @import("./layer.zig").Layer;
pub const DenseLayer = @import("./layer.zig").DenseLayer;

// pub fn add(a: i32, b: i32) i32 {
//     log.debug("add() called", .{});
//     return a + b;
// }
