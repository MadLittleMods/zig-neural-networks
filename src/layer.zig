const std = @import("std");

// Interface based off of https://www.openmymind.net/Zig-Interfaces/
pub const Layer = struct {
    // These two fields are the same as before
    ptr: *anyopaque,
    forwardFn: *const fn (
        ptr: *anyopaque,
        inputs: []const f64,
        allocator: std.mem.Allocator,
    ) anyerror![]f64,
    backwardFn: *const fn (
        ptr: *anyopaque,
        output_gradients: []const f64,
        allocator: std.mem.Allocator,
    ) anyerror![]f64,

    // This is new
    fn init(ptr: anytype) @This() {
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
                // `@call(.always_inline, ptr_info.Pointer.child.forward, .{self, data});`
                return ptr_info.Pointer.child.forward(self, inputs, allocator);
            }
            pub fn backward(
                pointer: *anyopaque,
                output_gradients: []const f64,
                allocator: std.mem.Allocator,
            ) anyerror![]f64 {
                const self: T = @ptrCast(@alignCast(pointer));
                return ptr_info.Pointer.child.backward(self, output_gradients, allocator);
            }
        };

        return .{
            .ptr = ptr,
            .forwardFn = gen.forward,
            .backwardFn = gen.backward,
        };
    }

    pub fn forward(
        self: @This(),
        inputs: []const f64,
        allocator: std.mem.Allocator,
    ) ![]f64 {
        return self.forwardFn(self.ptr, inputs, allocator);
    }

    pub fn backward(
        self: @This(),
        output_gradients: []const f64,
        allocator: std.mem.Allocator,
    ) ![]f64 {
        return self.backwardFn(self.ptr, output_gradients, allocator);
    }
};

/// "Dense" just means that every input is connected to every output. This is a normal
//neural network layer.
pub const DenseLayer = struct {
    // num_input_nodes: usize,
    // num_output_nodes: usize,
    // // Weights for each incoming connection. Each node in this layer has a weighted
    // // connection to each node in the previous layer (num_input_nodes * num_output_nodes).
    // //
    // // The weights are stored in row-major order where each row is the incoming
    // // connection weights for a single node in this layer.
    // // Size: num_output_nodes * num_input_nodes
    // weights: []f64,
    // // Bias for each node in the layer (num_output_nodes)
    // // Size: num_output_nodes
    // biases: []f64,

    pub fn forward(self: *@This(), inputs: []const f64, allocator: std.mem.Allocator) ![]f64 {
        _ = allocator;
        _ = self;
        // TODO

        return inputs;
    }

    pub fn backward(self: *@This(), output_gradients: []const f64, allocator: std.mem.Allocator) ![]f64 {
        _ = allocator;
        _ = self;
        // TODO

        return output_gradients;
    }

    pub fn layer(self: *@This()) Layer {
        return Layer.init(self);
    }
};
