//! A layer that adds noise to the input to help reduce overfitting. This should
//! probably be the first layer in the network. Alternatively, you could add noise to
//! the data points directly.
//!
//! Note: This should only be used during the training phase. Do NOT use this during
//! testing because you don't want to throw off and muddy your actual input data.
//!
//! (inherits from `Layer`)
const std = @import("std");
const log = std.log.scoped(.zig_neural_networks);

const Layer = @import("zig-neural-networks").Layer;

// It's nicer to have a fixed seed so we can reproduce the same results.
const seed = 123;
var prng = std.rand.DefaultPrng.init(seed);
const random_instance = prng.random();

// pub const CustomNoiseLayer = struct {
const Self = @This();

/// Hyper parameters (as opposed to normal parameters) are settings or configurations
/// that are set before the training process begins and are not updated during training.
pub const HyperParameters = struct {
    noise_probability: f64,
    noise_strength: f64,
};

hyper_parameters: HyperParameters,

pub fn init(
    /// Probability that noise is added to the input [0-1].
    /// This is probably best set to a low number like `0.01`
    noise_probability: f64,
    /// A scalar strength value multiplied by the noise. This can practically be any
    /// value but (0-1] would be a normal range. 0 would have the same effect of no
    /// noise.
    noise_strength: f64,
) !Self {
    return Self{
        .hyper_parameters = .{
            .noise_probability = noise_probability,
            .noise_strength = noise_strength,
        },
    };
}

pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
    _ = allocator;

    // This isn't strictly necessary but it marks the memory as dirty (010101...) in
    // safe modes (https://zig.news/kristoff/what-s-undefined-in-zig-9h)
    self.* = undefined;
}

/// Add some noise to `inputs` and return the outputs.
pub fn forward(
    self: *@This(),
    inputs: []const f64,
    allocator: std.mem.Allocator,
) ![]f64 {
    var outputs = try allocator.alloc(f64, inputs.len);
    for (inputs, 0..) |input, index| {
        var noise_value: f64 = 0.0;
        if (random_instance.float(f64) < self.hyper_parameters.noise_probability) {
            // Depending on the application, you may want noise between [0-1) (for
            // normalized images like the MNIST example) or a normal gaussian
            // distribution (for other TODO).
            //
            // Here we have a random value centered on 0, from [-0.5, 0.5) so we can
            // affect the pixel in both directions.
            const random_value = random_instance.float(f64) - 0.5;
            noise_value = self.hyper_parameters.noise_strength * random_value;
        }

        outputs[index] = std.math.clamp(input + noise_value, 0, 1);
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
    _ = self;
    // This is just a noop because we're the ones essentially creating the new inputs
    // (adding noise the inputs) (TODO: better way to explain).
    var output_gradient_copy = try allocator.alloc(f64, output_gradient.len);
    @memcpy(output_gradient_copy, output_gradient);
    return output_gradient_copy;
}

/// There are no parameters we need to update in an noise layer so this is just a
/// no-op. This layer only has hyper parameters (as opposed to normal parameters) which
/// are settings or configurations that are set before the training process begins and
/// are not updated during training.
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
    // What we output here, aligns with `Layer.SerializedLayer`. It's easier to use an
    // anonymous struct here instead of the `Layer.SerializedLayer` type because we know
    // the concrete type of the parameters here vs the generic `std.json.Value` from
    // `Layer.SerializedLayer`.
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

    const noise_layer = try init(
        hyper_parameters.noise_probability,
        hyper_parameters.noise_strength,
    );

    return noise_layer;
}
