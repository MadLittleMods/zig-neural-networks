//! Used in tests as a dummy custom layer someone might write. Just multiplies each
//! input by a scalar value
const std = @import("std");
const Layer = @import("../layers/Layer.zig");

// const CustomScalarLayer = struct {
const Self = @This();

pub const HyperParameters = struct {
    scalar: f64,
};

hyper_parameters: HyperParameters,

pub fn init(
    /// Just some arbitrary value that makes this layer unique.
    scalar: f64,
) !Self {
    return Self{
        .hyper_parameters = .{
            .scalar = scalar,
        },
    };
}

pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
    _ = allocator;

    // This isn't strictly necessary but it marks the memory as dirty (010101...) in
    // safe modes (https://zig.news/kristoff/what-s-undefined-in-zig-9h)
    self.* = undefined;
}

/// Multiply each input by the scalar
pub fn forward(
    self: *@This(),
    inputs: []const f64,
    allocator: std.mem.Allocator,
) ![]f64 {
    var outputs = try allocator.alloc(f64, inputs.len);
    for (inputs, outputs) |input, *output| {
        output.* = self.hyper_parameters.scalar * input;
    }

    return outputs;
}

/// Given the the derivative of the cost/loss/error with respect to the *outputs* (dC/dy),
/// returns the derivative of the cost/loss/error with respect to the *inputs* (dC/dx).
pub fn backward(
    self: *@This(),
    /// The partial derivative of the cost/loss/error with respect to the *outputs*
    /// (dC/dy).
    ///
    /// Size: num_output_nodes
    output_gradient: []const f64,
    allocator: std.mem.Allocator,
) ![]f64 {
    // `input_gradient` stores the partial derivative of the cost (C) with respect
    // to the inputs (x) -> (dC/dx).
    // dC/dx = dC/dy * dy/dx
    // (dy/dx) = self.hyper_parameters.scalar
    var input_gradient = try allocator.alloc(f64, output_gradient.len);
    for (output_gradient, input_gradient) |output_element, *input_element| {
        input_element.* = output_element * self.hyper_parameters.scalar;
    }
    return input_gradient;
}

/// There are no parameters we need to update in a CustomScalarLayer so this is just
/// a no-op.
pub fn applyCostGradients(self: *Self, learn_rate: f64, options: Layer.ApplyCostGradientsOptions) void {
    _ = self;
    _ = learn_rate;
    _ = options;
}

/// Helper to create a generic `Layer` that we can use in a `NeuralNetwork`
pub fn layer(self: *@This()) Layer {
    return Layer.init(self);
}

/// Serialize the layer to JSON (using the `std.json` library).
pub fn jsonStringify(self: @This(), jws: anytype) !void {
    try jws.write(.{
        .serialized_type_name = @typeName(Self),
        .parameters = self.hyper_parameters,
    });
}

/// Deserialize the layer from JSON (using the `std.json` library).
pub fn jsonParse(allocator: std.mem.Allocator, source: anytype, options: std.json.ParseOptions) !@This() {
    const json_value = try std.json.parseFromTokenSourceLeaky(std.json.Value, allocator, source, options);
    return try jsonParseFromValue(allocator, json_value, options);
}

/// Deserialize the layer from a parsed JSON value. (using the `std.json` library).
pub fn jsonParseFromValue(allocator: std.mem.Allocator, source: std.json.Value, options: std.json.ParseOptions) !@This() {
    const parsed_parameters = try std.json.parseFromValue(
        HyperParameters,
        allocator,
        source,
        options,
    );
    defer parsed_parameters.deinit();
    const hyper_parameters = parsed_parameters.value;

    const custom_scalar_layer = try init(
        hyper_parameters.scalar,
    );

    return custom_scalar_layer;
}
