//! A layer that applies an activation function to its inputs. The idiomatic way to use
//! these is to place them between every `DenseLayer`.
//!
//! (inherits from `Layer`)
const std = @import("std");
const log = std.log.scoped(.zig_neural_networks);

const Layer = @import("Layer.zig");
const ActivationFunction = @import("../activation_functions.zig").ActivationFunction;

// pub const ActivationLayer = struct {
const Self = @This();

pub const Parameters = struct {
    activation_function: ActivationFunction,
};

parameters: Parameters,

/// Store any inputs we get during the forward pass so we can use them during
/// the backward pass.
inputs: []const f64 = undefined,
pub fn init(activation_function: ActivationFunction) !Self {
    return Self{
        .parameters = .{
            .activation_function = activation_function,
        },
    };
}

pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
    _ = allocator;

    // This isn't strictly necessary but it marks the memory as dirty (010101...) in
    // safe modes (https://zig.news/kristoff/what-s-undefined-in-zig-9h)
    self.* = undefined;
}

pub fn forward(
    self: *@This(),
    inputs: []const f64,
    allocator: std.mem.Allocator,
) ![]f64 {
    // Store the inputs so we can use them during the backward pass.
    self.inputs = inputs;

    var outputs = try allocator.alloc(f64, inputs.len);
    for (inputs, 0..) |_, index| {
        outputs[index] = self.parameters.activation_function.activate(inputs, index);
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
    if (output_gradient.len != self.inputs.len) {
        log.err("ActivationLayer.backward() was called with a output_gradient of length {d} " ++
            "but we expect it to match the same length as the inputs we saw during the foward pass {d}", .{
            output_gradient.len,
            self.inputs.len,
        });

        return error.OutputGradientLengthMismatch;
    }

    // `input_gradient` stores the partial derivative of the cost (C) with respect
    // to the inputs (x) -> (dC/dx).
    // dC/dx = dC/dy * dy/dx
    // (dy/dx) = activation_function_derivative
    var input_gradient = try allocator.alloc(f64, output_gradient.len);

    // Calculate the change of the activation function (y) with respect to the
    // input of each node ("change of" is just another way to say "derivative
    // of")
    //
    // After we find the derivative of the activation function with respect to the
    // input of each node, we can multiply/dot it with the derivative of the
    // cost function with respect to the activation output of the same node to
    // produce the `input_gradient` derivative for each node.
    for (0..self.inputs.len) |index| {
        // Check if we can do an efficient shortcut in these calculations (depends
        // on the activation function)
        //
        // See the [developer notes on the activation
        // functions](../../dev-notes.md#activation-functions) to understand why we
        // do this.
        switch (self.parameters.activation_function.hasSingleInputActivationFunction()) {
            // If the activation function (y) only uses a single input to produce an
            // output, the "derivative" of the activation function will result in a
            // sparse Jacobian matrix with only the diagonal elements populated (and
            // the rest 0). And we can do an efficient shortcut in the calculations.
            //
            // Sparse Jacobian matrix where only the diagonal elements are defined:
            // ┏  𝝏y_1   0     0     0    ┓
            // ┃  𝝏x_1                    ┃
            // ┃                          ┃
            // ┃   0    𝝏y_2   0     0    ┃
            // ┃        𝝏x_2              ┃
            // ┃                          ┃
            // ┃   0     0    𝝏y_3   0    ┃
            // ┃              𝝏x_3        ┃
            // ┃                          ┃
            // ┃   0     0     0    𝝏y_4  ┃
            // ┗                    𝝏x_4  ┛
            //
            // If we think about doing the dot product between cost derivatives
            // vector and each row of this sparse Jacobian matrix, we can see that
            // we only end up with the diagonal elements multiplied by the other
            // vector and the rest fall away because they are multiplied by 0.
            //
            // ┏  𝝏y_1   0     0     0    ┓     ┏  𝝏C    ┓
            // ┃  𝝏x_1                    ┃     ┃  𝝏y_1  ┃
            // ┃                          ┃     ┃        ┃
            // ┃   0    𝝏y_2   0     0    ┃     ┃  𝝏C    ┃
            // ┃        𝝏x_2              ┃     ┃  𝝏y_2  ┃
            // ┃                          ┃  .  ┃        ┃  = 𝝏C (input_gradient)
            // ┃   0     0    𝝏y_3   0    ┃     ┃  𝝏C    ┃    𝝏x
            // ┃              𝝏x_3        ┃     ┃  𝝏y_3  ┃
            // ┃                          ┃     ┃        ┃
            // ┃   0     0     0    𝝏y_4  ┃     ┃  𝝏C    ┃
            // ┗                    𝝏x_4  ┛     ┗  𝝏y_4  ┛
            //
            // For example to calculate `input_gradient[0]`,
            // it would look like:
            // input_gradient[0] = 𝝏y_1 * 𝝏C    +  0 * 𝝏C    +  0 * 𝝏C    +  0 * 𝝏C
            //                     𝝏x_1   𝝏y_1         𝝏y_2         𝝏y_3         𝝏y_4
            //
            //                   = 𝝏y_1 * 𝝏C
            //                     𝝏x_1   𝝏y_1
            //
            // Since all of those extra multiplictions fall away against the sparse
            // matrix anyway, to avoid the vector/matrix multiplication
            // computational complexity, we can see that we only need find the
            // partial derivative of the activation function with respect to the
            // input of the current node and multiply it with the partial
            // derivative of the cost with respect to the activation output of the
            // same node (where `k = i`).
            true => {
                // The partial derivative of activation function (y) with
                // respect to the input (x) -> (dy/dx)
                //
                // dy/dx = activation_function.derivative(x)
                const activation_derivative = self.parameters.activation_function.derivative(
                    self.inputs,
                    index,
                );
                input_gradient[index] = output_gradient[index] * activation_derivative;
            },
            // If the activation function (y_i) uses multiple inputs to produce an
            // output, the "derivative" of the activation function will result in a
            // full Jacobian matrix that we carefully have to matrix multiply with
            // the cost derivatives vector.
            //
            // ┏  𝝏y_1  𝝏y_1  𝝏y_1  𝝏y_1  ┓     ┏  𝝏C    ┓
            // ┃  𝝏x_1  𝝏x_2  𝝏x_3  𝝏x_4  ┃     ┃  𝝏y_1  ┃
            // ┃                          ┃     ┃        ┃
            // ┃  𝝏y_2  𝝏y_2  𝝏y_2  𝝏y_2  ┃     ┃  𝝏C    ┃
            // ┃  𝝏x_1  𝝏x_2  𝝏x_3  𝝏x_4  ┃     ┃  𝝏y_2  ┃
            // ┃                          ┃  .  ┃        ┃  = 𝝏C (input_gradient)
            // ┃  𝝏y_3  𝝏y_3  𝝏y_3  𝝏y_3  ┃     ┃  𝝏C    ┃    𝝏x
            // ┃  𝝏x_1  𝝏x_2  𝝏x_3  𝝏x_4  ┃     ┃  𝝏y_3  ┃
            // ┃                          ┃     ┃        ┃
            // ┃  𝝏y_4  𝝏y_4  𝝏y_4  𝝏y_4  ┃     ┃  𝝏C    ┃
            // ┗  𝝏x_1  𝝏x_2  𝝏x_3  𝝏x_4  ┛     ┗  𝝏y_4  ┛
            //
            // For example to calculate `input_gradient[0]`,
            // it would look like:
            // input_gradient[0] = 𝝏y_1 * 𝝏C    +  𝝏y_1 * 𝝏C    +  𝝏y_1 * 𝝏C    +  𝝏y_1 * 𝝏C
            //                     𝝏x_1   𝝏y_1     𝝏x_2   𝝏y_2     𝝏x_3   𝝏y_3     𝝏x_4   𝝏y_4
            //
            // Since we only work on one output node at a time, we just take it row
            // by row on the matrix and do the dot product with the cost derivatives
            // vector.
            //
            // Note: There are more efficient ways to do this type of calculation
            // when working directly with the matrices but this seems like the most
            // beginner friendly way to do it and in the spirit of the other code.
            false => {
                // For each node, find the partial derivative of activation
                // function (y_i) with respect to the input (x_k) of that node.
                // We're basically just producing row j (where j = `node_index`)
                // of the Jacobian matrix for the activation function.
                //
                // dy/dx = activation_function.jacobian_row(x)
                const activation_ki_derivatives = try self.parameters.activation_function.jacobian_row(
                    self.inputs,
                    index,
                    allocator,
                );
                defer allocator.free(activation_ki_derivatives);

                // This is just a dot product of the `activation_ki_derivatives`
                // and `output_gradient` (both vectors). Or can also be thought
                // of as a matrix multiplication between a 1xn matrix
                // (activation_ki_derivatives) and a nx1 matrix
                // (output_gradient).
                for (activation_ki_derivatives, 0..) |_, activation_derivative_index| {
                    input_gradient[index] += activation_ki_derivatives[activation_derivative_index] *
                        output_gradient[activation_derivative_index];
                }
            },
        }
    }

    return input_gradient;
}

/// There are no parameters we need to update in an activation layer so this
/// is just a no-op.
pub fn applyCostGradients(self: *Self, learn_rate: f64, options: Layer.ApplyCostGradientsOptions) void {
    _ = self;
    _ = learn_rate;
    _ = options;
}

/// Helper to create a generic `Layer` that we can use in a `NerualNetwork`
pub fn layer(self: *@This()) Layer {
    return Layer.init(self);
}

pub fn serialize(self: *@This(), allocator: std.mem.Allocator) ![]const u8 {
    const json_text = try std.json.stringifyAlloc(
        allocator,
        self.parameters,
        .{
            .whitespace = .indent_2,
        },
    );
    return json_text;
}

/// Turn some serialized parameters back into a `ActivationLayer`.
pub fn deserialize(self: *@This(), json: std.json.Value, allocator: std.mem.Allocator) !void {
    const parsed_parameters = try std.json.parseFromValue(
        Parameters,
        allocator,
        json,
        .{},
    );
    defer parsed_parameters.deinit();
    const parameters = parsed_parameters.value;

    const activation_layer = try init(
        parameters.activation_function,
    );

    self.* = activation_layer;
}

/// Serialize the layer to JSON (using the `std.json` library).
pub fn jsonStringify(self: @This(), jws: anytype) !void {
    // What we output here, aligns with `Layer.SerializedLayer`. It's easier to use an
    // anonymous struct here instead of the `Layer.SerializedLayer` type because we know
    // the concrete type of the parameters here vs the generic `std.json.Value` from
    // `Layer.SerializedLayer`. Plus it's just more boilerplate for us to get
    // `self.parameters` into `std.json.Value` if we went that route.
    try jws.write(.{
        .serialized_type_name = @typeName(Self),
        .parameters = self.parameters,
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
        Parameters,
        allocator,
        source,
        options,
    );
    defer parsed_parameters.deinit();
    const parameters = parsed_parameters.value;

    const activation_layer = try init(
        parameters.activation_function,
        allocator,
    );

    return activation_layer;
}
