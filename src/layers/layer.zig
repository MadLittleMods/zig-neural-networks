const std = @import("std");

/// Neural Network `Layer` base class
//
// Interface implementation based off of https://www.openmymind.net/Zig-Interfaces/
pub const Layer = struct {
    ptr: *anyopaque,
    forwardFn: *const fn (
        ptr: *anyopaque,
        inputs: []const f64,
        allocator: std.mem.Allocator,
    ) anyerror![]f64,
    backwardFn: *const fn (
        ptr: *anyopaque,
        output_gradient: []const f64,
        learn_rate: f64,
        allocator: std.mem.Allocator,
    ) anyerror![]f64,

    /// A generic constructor that any sub-classes can use to create a `Layer`.
    //
    // All of this complexity here allows the sub-classes to stand on their own instead
    // of having to deal with awkward member functions that take `ptr: *anyopaque` which
    // we can't call directly. See the "Making it Prettier" section in
    // https://www.openmymind.net/Zig-Interfaces/.
    fn init(
        /// Because of the `anytype` here, all of this runs at comptime
        ptr: anytype,
    ) @This() {
        const T = @TypeOf(ptr);
        const ptr_info = @typeInfo(T);

        if (ptr_info != .Pointer) @compileError("ptr must be a pointer");
        if (ptr_info.Pointer.size != .One) @compileError("ptr must be a single item pointer");

        const gen = struct {
            pub fn forward(
                pointer: *anyopaque,
                inputs: []const f64,
                allocator: std.mem.Allocator,
            ) anyerror![]f64 {
                const self: T = @ptrCast(@alignCast(pointer));
                // We could alternatively use
                // `@call(.always_inline, ptr_info.Pointer.child.forward, .{ self, inputs, allocator });`
                // for any of these functions calls. It would be best to test if this has
                // a performance impact before making our code more complex though.
                return ptr_info.Pointer.child.forward(self, inputs, allocator);
            }
            pub fn backward(
                pointer: *anyopaque,
                output_gradient: []const f64,
                learn_rate: f64,
                allocator: std.mem.Allocator,
            ) anyerror![]f64 {
                _ = learn_rate;
                const self: T = @ptrCast(@alignCast(pointer));
                return ptr_info.Pointer.child.backward(self, output_gradient, allocator);
            }
        };

        return .{
            .ptr = ptr,
            .forwardFn = gen.forward,
            .backwardFn = gen.backward,
        };
    }

    /// TODO: write description
    pub fn forward(
        self: @This(),
        inputs: []const f64,
        allocator: std.mem.Allocator,
    ) ![]f64 {
        return self.forwardFn(self.ptr, inputs, allocator);
    }

    /// This function is responsible for updating any parameters in the layer (e.g.
    /// weights and biases) and returns the derivative of the loss/cost/error with
    /// respect to the inputs.
    pub fn backward(
        self: @This(),
        /// The derivative of the loss/cost/error with respect to the outputs.
        output_gradient: []const f64,
        learn_rate: f64,
        allocator: std.mem.Allocator,
    ) ![]f64 {
        _ = learn_rate;
        return self.backwardFn(self.ptr, output_gradient, allocator);
    }
};
