//! Neural Network `Layer` interface (base class)
//!
//! Usage: Any struct that has a `deinit`, `forward`, `backward`, and
//! `applyCostGradients` function can be used as a `Layer`. In your custom layer struct,
//! it's also best to provide a helper function to create the generic `Layer` struct:
//!
//! ```
//! pub fn layer(self: *@This()) Layer {
//!     return Layer.init(self);
//! }
//! ```
const std = @import("std");

pub const ApplyCostGradientsOptions = struct {
    /// See the comment in `NerualNetwork.learn()` for more info
    momentum: f64 = 0,
};

// Interface implementation based off of https://www.openmymind.net/Zig-Interfaces/
// pub const Layer = struct {
ptr: *anyopaque,
deinitFn: *const fn (
    ptr: *anyopaque,
    allocator: std.mem.Allocator,
) void,
forwardFn: *const fn (
    ptr: *anyopaque,
    inputs: []const f64,
    allocator: std.mem.Allocator,
) anyerror![]f64,
backwardFn: *const fn (
    ptr: *anyopaque,
    output_gradient: []const f64,
    allocator: std.mem.Allocator,
) anyerror![]f64,
applyCostGradientsFn: *const fn (
    ptr: *anyopaque,
    learn_rate: f64,
    options: ApplyCostGradientsOptions,
) void,

/// A generic constructor that any sub-classes can use to create a `Layer`.
//
// All of this complexity here allows the sub-classes to stand on their own instead
// of having to deal with awkward member functions that take `ptr: *anyopaque` which
// we can't call directly. See the "Making it Prettier" section in
// https://www.openmymind.net/Zig-Interfaces/.
pub fn init(
    /// Because of the `anytype` here, all of this runs at comptime
    ptr: anytype,
) @This() {
    const T = @TypeOf(ptr);
    const ptr_info = @typeInfo(T);

    if (ptr_info != .Pointer) @compileError("ptr must be a pointer");
    if (ptr_info.Pointer.size != .One) @compileError("ptr must be a single item pointer");

    const gen = struct {
        pub fn deinit(
            pointer: *anyopaque,
            allocator: std.mem.Allocator,
        ) void {
            const self: T = @ptrCast(@alignCast(pointer));
            ptr_info.Pointer.child.deinit(self, allocator);
        }
        pub fn forward(
            pointer: *anyopaque,
            inputs: []const f64,
            allocator: std.mem.Allocator,
        ) anyerror![]f64 {
            const self: T = @ptrCast(@alignCast(pointer));
            // We could alternatively use `@call(.always_inline,
            // ptr_info.Pointer.child.forward, .{ self, inputs, allocator });` for
            // any of these functions calls. It would be best to test if this has a
            // performance impact before making our code more complex though. Using
            // this also has the benefit of cleaning up the call-stack.
            return ptr_info.Pointer.child.forward(self, inputs, allocator);
        }
        pub fn backward(
            pointer: *anyopaque,
            output_gradient: []const f64,
            allocator: std.mem.Allocator,
        ) anyerror![]f64 {
            const self: T = @ptrCast(@alignCast(pointer));
            return ptr_info.Pointer.child.backward(self, output_gradient, allocator);
        }
        pub fn applyCostGradients(
            pointer: *anyopaque,
            learn_rate: f64,
            options: ApplyCostGradientsOptions,
        ) void {
            const self: T = @ptrCast(@alignCast(pointer));
            return ptr_info.Pointer.child.applyCostGradients(self, learn_rate, options);
        }
    };

    return .{
        .ptr = ptr,
        .deinitFn = gen.deinit,
        .forwardFn = gen.forward,
        .backwardFn = gen.backward,
        .applyCostGradientsFn = gen.applyCostGradients,
    };
}

/// Used to clean-up any allocated resources used in the layer.
pub fn deinit(self: @This(), allocator: std.mem.Allocator) void {
    return self.deinitFn(self.ptr, allocator);
}

/// Run the given `inputs` through a layer of the neural network and return the outputs.
pub fn forward(
    self: @This(),
    inputs: []const f64,
    allocator: std.mem.Allocator,
) ![]f64 {
    return self.forwardFn(self.ptr, inputs, allocator);
}

/// Given the the derivative of the cost/loss/error with respect to the *outputs*
/// (dC/dy), returns the derivative of the cost/loss/error with respect to the
/// *inputs* (dC/dx). This function is also responsible for updating any parameters
/// in the layer (e.g. weights and biases).
pub fn backward(
    self: @This(),
    /// The derivative of the cost/loss/error with respect to the outputs.
    output_gradient: []const f64,
    allocator: std.mem.Allocator,
) ![]f64 {
    return self.backwardFn(self.ptr, output_gradient, allocator);
}

/// This function is responsible for updating any parameters in the layer (e.g.
/// weights and biases). Also should reset the cost gradients back to zero.
pub fn applyCostGradients(
    self: @This(),
    /// This configures how big of a step we make down the cost gradient landscape. This
    /// is usually a small value like `0.1` or `0.05` because large values can cause us
    /// to skip past the valley/hole where the minimum value of the cost is.
    learn_rate: f64,
    options: ApplyCostGradientsOptions,
) void {
    self.applyCostGradientsFn(self.ptr, learn_rate, options);
}
