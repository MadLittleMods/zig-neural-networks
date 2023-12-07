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
const log = std.log.scoped(.zig_neural_networks);
const DenseLayer = @import("./DenseLayer.zig");
const ActivationLayer = @import("./ActivationLayer.zig");

// Just trying to copy whatever `std.json.stringifyAlloc` does because we can't use
// `anytype` in a function pointer definition
const WriteStream = std.json.WriteStream(
    std.ArrayList(u8).Writer,
    .{ .checked_to_arbitrary_depth = {} },
);

pub const ApplyCostGradientsOptions = struct {
    /// See the comment in `NerualNetwork.learn()` for more info
    momentum: f64 = 0,
};

const Self = @This();

const JsonDeserializeFn = *const fn (
    allocator: std.mem.Allocator,
    source: std.json.Value,
) std.json.ParseFromValueError!Self;
/// The layers already known to the library
const builtin_type_name_to_deserialize_layer_fn_map = std.ComptimeStringMap(JsonDeserializeFn, .{
    .{ @typeName(DenseLayer), deserializeFnFromLayer(DenseLayer) },
    .{ @typeName(ActivationLayer), deserializeFnFromLayer(ActivationLayer) },
});
/// Stores the custom layer types that people can register. Basically acts as mutable
/// namespaced global state. We could make it `pub` to allow people to interact directly
/// but we prefer people just to use the helper functions.
var type_name_to_deserialize_layer_fn_map: std.StringHashMapUnmanaged(JsonDeserializeFn) = .{};
/// Register a custom layer type so that it can be deserialized from JSON.
pub fn registerCustomLayer(comptime T: type, allocator: std.mem.Allocator) !void {
    try type_name_to_deserialize_layer_fn_map.put(
        allocator,
        @typeName(T),
        deserializeFnFromLayer(T),
    );
}
/// De-initialize the custom layer type map (needs to be called if `registerCustomLayer`
/// is used).
pub fn deinitCustomLayerMap(allocator: std.mem.Allocator) void {
    type_name_to_deserialize_layer_fn_map.deinit(allocator);
}

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
jsonStringifyFn: *const fn (
    ptr: *anyopaque,
    jws: *WriteStream,
) error{OutOfMemory}!void,

/// A generic constructor that any sub-classes can use to create a `Layer`.
//
// All of this complexity here allows the sub-classes to stand on their own instead
// of having to deal with awkward member functions that take `ptr: *anyopaque` which
// we can't call directly. See the "Making it Prettier" section in
// https://www.openmymind.net/Zig-Interfaces/.
pub fn init(
    /// Because of the `anytype` here, all of this runs at comptime
    ptr: anytype,
) Self {
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
        pub fn jsonStringify(pointer: *anyopaque, jws: *WriteStream) error{OutOfMemory}!void {
            const self: T = @ptrCast(@alignCast(pointer));
            return try ptr_info.Pointer.child.jsonStringify(self.*, jws);
        }
    };

    return .{
        .ptr = ptr,
        .deinitFn = gen.deinit,
        .forwardFn = gen.forward,
        .backwardFn = gen.backward,
        .applyCostGradientsFn = gen.applyCostGradients,
        .jsonStringifyFn = gen.jsonStringify,
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

/// Serialize the layer to JSON (using the `std.json` library).
pub fn jsonStringify(self: @This(), jws: *WriteStream) !void {
    return try self.jsonStringifyFn(self.ptr, jws);
}

const SerializedLayer = struct {
    serialized_type_name: []const u8,
    parameters: std.json.Value,
};

/// Deserialize the layer from JSON (using the `std.json` library).
pub fn jsonParse(allocator: std.mem.Allocator, source: anytype, options: std.json.ParseOptions) !@This() {
    const json_value = try std.json.parseFromTokenSourceLeaky(std.json.Value, allocator, source, options);
    return try jsonParseFromValue(allocator, json_value, options);
}

/// Deserialize the layer from a parsed JSON value. (using the `std.json` library).
pub fn jsonParseFromValue(allocator: std.mem.Allocator, source: std.json.Value, options: std.json.ParseOptions) !@This() {
    const parsed_serialized_layer = try std.json.parseFromValue(
        SerializedLayer,
        allocator,
        source,
        options,
    );
    defer parsed_serialized_layer.deinit();
    const serialized_layer = parsed_serialized_layer.value;

    const deserializeFn =
        // First check the built-in types since those are probably the most common
        // anyway and since we're using a `std.ComptimeStringMap`, should have a faster lookup
        builtin_type_name_to_deserialize_layer_fn_map.get(serialized_layer.serialized_type_name) orelse
        // Then check the custom layer types that people can register
        type_name_to_deserialize_layer_fn_map.get(serialized_layer.serialized_type_name) orelse {
        log.err("Unknown serialized_type_name {s} (does not match any known layer types). " ++
            "Try making the library aware of this custom layer type with " ++
            "`Layer.registerCustomLayer({0s}, allocator)`", .{
            serialized_layer.serialized_type_name,
        });
        return std.json.ParseFromValueError.UnknownField;
    };
    const generic_layer = deserializeFn(
        allocator,
        serialized_layer.parameters,
    ) catch |err| {
        // We use a `catch` here to give some sane info and context
        log.err("Unable to deserialize {s} with {any}. Error from deserialize() -> {any}", .{
            serialized_layer.serialized_type_name,
            serialized_layer.parameters,
            err,
        });
        return err;
    };

    return generic_layer;
}

/// Helper to create a `JsonDeserializeFn` for a specific layer type
pub fn deserializeFnFromLayer(comptime T: type) JsonDeserializeFn {
    const gen = struct {
        pub fn deserialize(
            allocator: std.mem.Allocator,
            source: std.json.Value,
        ) std.json.ParseFromValueError!Self {
            var specific_layer = try T.jsonParseFromValue(allocator, source, .{});
            return specific_layer.layer();
        }
    };

    return gen.deserialize;
}
