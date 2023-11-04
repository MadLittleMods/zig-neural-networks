const std = @import("std");

pub const ApplyCostGradientsOptions = struct {
    // The momentum to apply to gradient descent. This is a value between 0 and 1
    // and often has a value close to 1.0, such as 0.8, 0.9, or 0.99. A momentum of
    // 0.0 is the same as gradient descent without momentum.
    //
    // Momentum is used to help the gradient descent algorithm keep the learning
    // process going in the right direction between different batches. It does this
    // by adding a fraction of the previous weight change to the current weight
    // change. Essentially, if it was moving before, it will keep moving in the same
    // direction. It's most useful in situations where the cost surface has lots of
    // curvature (changes a lot) ("highly non-spherical") or when the cost surface
    // "flat or nearly flat, e.g. zero gradient. The momentum allows the search to
    // progress in the same direction as before the flat spot and helpfully cross
    // the flat region."
    // (https://machinelearningmastery.com/gradient-descent-with-momentum-from-scratch/)
    //
    // > The momentum algorithm accumulates an exponentially decaying moving average
    // > of past gradients and continues to move in their direction.
    // >
    // > -- *Deep Learning* book page 296 (Ian Goodfellow)
    momentum: f64 = 0,
};

/// Neural Network `Layer` base class
//
// Interface implementation based off of https://www.openmymind.net/Zig-Interfaces/
pub const Layer = struct {
    ptr: *anyopaque,
    deinitFn: *const fn (
        ptr: *anyopaque,
        allocator: std.mem.Allocator,
    ) anyerror![]f64,
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
    fn init(
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
            ) anyerror![]f64 {
                const self: T = @ptrCast(@alignCast(pointer));
                return ptr_info.Pointer.child.deinit(self, allocator);
            }
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

    fn deinit(self: @This(), allocator: std.mem.Allocator) void {
        _ = allocator;
        _ = self;
    }

    /// TODO: write description
    pub fn forward(
        self: @This(),
        inputs: []const f64,
        allocator: std.mem.Allocator,
    ) ![]f64 {
        return self.forwardFn(self.ptr, inputs, allocator);
    }

    /// Given the the derivative of the cost/loss/error with respect to the *outputs*,
    /// returns the derivative of the cost/loss/error with respect to the *inputs*. Also
    /// responsible for updating any parameters in the layer (e.g. weights and biases).
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
        /// TODO: write description
        learn_rate: f64,
        options: ApplyCostGradientsOptions,
    ) void {
        self.applyCostGradientsFn(self.ptr, learn_rate, options);
    }
};
