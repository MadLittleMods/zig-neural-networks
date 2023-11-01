pub const activation_functions = @import("main.zig");

test {
    // https://ziglang.org/documentation/master/#Nested-Container-Tests
    @import("std").testing.refAllDecls(@This());
}
