pub const neural_network_tests = @import("./tests/neural_network_tests.zig");

pub const data_point = @import("data_point.zig");
pub const activation_functions = @import("activation_functions.zig");
pub const cost_functions = @import("cost_functions.zig");

test {
    // https://ziglang.org/documentation/master/#Nested-Container-Tests
    @import("std").testing.refAllDecls(@This());
}
