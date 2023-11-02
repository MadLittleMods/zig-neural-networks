const std = @import("std");

// Activation functions allows a layer to have a non-linear affect on the output so they
// can bend the boundary around the data. Without this, the network would only be able
// to separate data with a straight line.
//
// The choice of activation also doesn't matter *that* much. If the network training is
// unstable, you can usually fix that with a choice of hyperparameters at maybe a 2-3x
// cost in speed. You usually don't have a huge bump in the final accuracy you achieve.
// Things other than activations usually dominate the cost.

// TODO: In the future, we could add Sigmoid, Tanh, SiLU, etc to try out

// ReLU (Rectified linear unit)
// TODO: Visualize this (ASCII art)
//
// ReLU, for some reason, seems to result in really choppy loss surfaces that make it
// finicky to get the network to converge without a lot of hyperparameter tuning. The
// fact that ReLU never outputs negatives and has huge regions with a gradient of 0
// causes problems. Any time you would reach for ReLU, you're probably better off with
// one of the alternatives like Leaky ReLU or ELU.
//
// ReLU enforces the inductive bias that small data should be ignored completely and
// that big data should be propagated without changes (think about a dim light in a dark
// room; the dark regions are just noise, no matter what your eyes think they perceive,
// but the dim light carries information, and further processing should be presented
// exactly that same information).
//
// ReLU was popularized as one of the first activations which was efficiently
// implemented on GPUs and didn't have vanishing/exploding gradient issues in deep
// networks.
//
// > The operation of ReLU is closer to the way our biological neurons work [(naive
// > model of the visual cortex)].
// >
// > ReLU is non-linear and has the advantage of not having any backpropagation errors
// > unlike the sigmoid function
//
// https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/
pub const Relu = struct {
    pub const has_single_input_activation_function = true;

    pub fn activate(_: @This(), inputs: []const f64, input_index: usize) f64 {
        const input = inputs[input_index];
        return @max(0.0, input);

        // Or in other words:
        //
        // if (input > 0.0) {
        //     return input;
        // }
        // return 0.0;
    }

    pub fn derivative(_: @This(), inputs: []const f64, input_index: usize) f64 {
        const input = inputs[input_index];
        if (input > 0.0) {
            return 1.0;
        }
        return 0.0;
    }
};

// LeakyReLU
// TODO: Visualize this (ASCII art)
//
// Like ReLU except when x < 0, LeakyReLU will have a small negative slope instead of a
// hard zero. This slope is typically a small value like 0.01.
//
// > [This solves the Dying ReLU problem] we see in ReLU [...] where some ReLU Neurons
// > essentially die for all inputs and remain inactive no matter what input is
// > supplied, here no gradient flows and if large number of dead neurons are there in a
// > Neural Network it’s performance is affected, this can be corrected by making use of
// > what is called Leaky ReLU.
// >
// > -- https://himanshuxd.medium.com/activation-functions-sigmoid-relu-leaky-relu-and-softmax-basics-for-neural-networks-and-deep-8d9c70eed91e
pub const LeakyRelu = struct {
    pub const has_single_input_activation_function = true;
    const alpha = 0.1;

    pub fn activate(_: @This(), inputs: []const f64, input_index: usize) f64 {
        const input = inputs[input_index];
        if (input > 0.0) {
            return input;
        }
        return alpha * input;
    }

    pub fn derivative(_: @This(), inputs: []const f64, input_index: usize) f64 {
        const input = inputs[input_index];
        if (input > 0.0) {
            return 1.0;
        }
        return alpha;
    }
};

// ELU (Exponential Linear Unit)
//
// It looks like LeakyReLU except with a smooth transition in the corner instead of that
// sharp transition that asymotically approaches -1. Not having a corner in the function
// makes some training routines more stable, and even though there are few more
// calculations involed with ELU, the network is usually large enough that matrix
// multiplication dominates the CPU/time.
//
// Once a network is trained it's often easy to fine-tune it to use a more efficient
// activation function. You can, for example, train it with ELU (which may be more
// likely to succeed at the first pass), swap the activation with LeakyReLU and leave
// everything else alone, and re-train on the same data with a tiny fraction of the
// initial epochs. You now have a network that uses LeakyReLU for cheaper inference but
// which didn't suffer instability during training.
pub const ELU = struct {
    pub const has_single_input_activation_function = true;
    const alpha = 1.0;

    pub fn activate(_: @This(), inputs: []const f64, input_index: usize) f64 {
        const input = inputs[input_index];
        if (input > 0.0) {
            return input;
        }

        return alpha * (@exp(input) - 1.0);
    }

    pub fn derivative(_: @This(), inputs: []const f64, input_index: usize) f64 {
        const input = inputs[input_index];
        if (input > 0.0) {
            return 1.0;
        }

        return alpha * @exp(input);
    }
};

// Sigmoid
// TODO: Visualize this (ASCII art)
// TODO: Why would someone use this one?
//
// Sigmoid will constrain things between 0 and 1 and not have many values in between.
pub const Sigmoid = struct {
    pub const has_single_input_activation_function = true;

    pub fn activate(_: @This(), inputs: []const f64, input_index: usize) f64 {
        const input = inputs[input_index];
        return 1.0 / (1.0 + @exp(-input));
    }

    pub fn derivative(self: @This(), inputs: []const f64, input_index: usize) f64 {
        const activation_value = self.activate(inputs, input_index);
        return activation_value * (1.0 - activation_value);
    }
};

// SoftMax squishes the output between [0, 1] and all the resulting elements add up to
// 1. So in terms of usage, this function will tell you what percentage that the
// given value at the `input_index` makes up the total sum of all the values in the
// array.

/// SoftMax is basically the multi-dimensional version of Sigmoid. See the [developer
/// notes on SoftMax](../../dev-notes.md) to see how the equation is derived.
//
// TODO: Visualize this (ASCII art)
// TODO: Why would someone use this one?
//
// Resources:
//  - Dahal, Paras. (Jun 2017). Softmax and Cross Entropy Loss. Paras Dahal.
//    https://parasdahal.com/softmax-crossentropy.
//  - Softmax Layer from Scratch | Mathematics & Python Code (by The Independent Code),
//    https://www.youtube.com/watch?v=AbLvJVwySEo
//  - https://themaverickmeerkat.com/2019-10-23-Softmax/
pub const SoftMax = struct {
    // SoftMax is not a single-input activation function. It uses all of the given
    // `inputs` to produce a single output. So we need to make sure to define a
    // `jacobian_row` function.
    pub const has_single_input_activation_function = false;

    pub fn activate(_: @This(), inputs: []const f64, input_index: usize) f64 {
        // TODO: Since it's really easy for the exponents to exceed the max value of a
        // float, we could subtract the max value from each input to make sure we don't
        // overflow (numerically stable): `@exp(input - max(inputs))` (do this if we ever
        // start seeing NaNs)

        var exp_sum: f64 = 0.0;
        for (inputs) |input| {
            exp_sum += @exp(input);
        }

        const exp_input = @exp(inputs[input_index]);

        // SoftMax equation: f(x) = e^x / Σ(e^x)
        return exp_input / exp_sum;
    }

    // Returns the partial derivative of the activation function with respect to the
    // input at the given index (x_k). This function only produces the diagonal elements
    // of the Jacobian matrix (where k = i) of the row specified by `input_index`.
    //
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
    // This is only defined for completeness sake but backpropagation should use the
    // `jacobian_row` function instead which calculates the actual derivative of the
    // activation function. Empirically (in practice), using the `derivative` function
    // will allow the neural network to converge successfully but it's unclear/not
    // measured on how much this causes us to wander around the cost/loss surface or
    // learn slower because it's flawed.
    pub fn derivative(_: @This(), inputs: []const f64, input_index: usize) f64 {
        var exp_sum: f64 = 0.0;
        for (inputs) |input| {
            exp_sum += @exp(input);
        }

        const exp_input = @exp(inputs[input_index]);

        // See the [developer notes on SoftMax](../../dev-notes.md#softmax) to
        // see how the equation is derived.
        return (exp_input * exp_sum - exp_input * exp_input) / (exp_sum * exp_sum);
    }

    // Returns all of the partial derivatives of the activation function (y_i) with
    // respect to the input at the given index (x_k). This function returns a single row
    // (specified by `row_index`) of the Jacobian matrix.
    //
    // ┏  𝝏y_1  𝝏y_1  𝝏y_1  𝝏y_1  ┓
    // ┃  𝝏x_1  𝝏x_2  𝝏x_3  𝝏x_4  ┃
    // ┃                          ┃
    // ┃  𝝏y_2  𝝏y_2  𝝏y_2  𝝏y_2  ┃
    // ┃  𝝏x_1  𝝏x_2  𝝏x_3  𝝏x_4  ┃
    // ┃                          ┃
    // ┃  𝝏y_3  𝝏y_3  𝝏y_3  𝝏y_3  ┃
    // ┃  𝝏x_1  𝝏x_2  𝝏x_3  𝝏x_4  ┃
    // ┃                          ┃
    // ┃  𝝏y_4  𝝏y_4  𝝏y_4  𝝏y_4  ┃
    // ┗  𝝏x_1  𝝏x_2  𝝏x_3  𝝏x_4  ┛
    //
    // Because the SoftMax activation function uses multiple inputs to produce a single
    // output, when we use a Jacobian matrix to find the derivative of the activation
    // function, all of elements will be defined (full, not sparse). This means that we
    // need to use the `jacobian_row()` function to accurately calculate the results
    // during backpropagation. Since this is SoftMax, this activation function will be
    // used as the final step of the output layer and will be dot producted with the
    // cost/loss/error vector to calculate the partial derivaties of the cost with
    // respect to the input (see `shareable_node_derivatives`).
    pub fn jacobian_row(
        _: @This(),
        inputs: []const f64,
        row_index: usize,
        allocator: std.mem.Allocator,
    ) ![]f64 {
        var results = try allocator.alloc(f64, inputs.len);

        var exp_sum: f64 = 0.0;
        for (inputs) |input| {
            exp_sum += @exp(input);
        }
        const denominator = exp_sum * exp_sum;

        const i = row_index;
        const exp_i = @exp(inputs[i]);
        for (inputs, 0..) |_, k| {
            const delta: f64 = if (i == k) 1 else 0;

            const exp_k = @exp(inputs[k]);

            // See the [developer notes on SoftMax](../../dev-notes.md#softmax) to
            // see how the equation is derived.
            const numerator = (delta * exp_i * exp_sum) - (exp_i * exp_k);

            const result_ki = numerator / denominator;
            results[k] = result_ki;
        }

        return results;
    }
};

/// Estimate the slope of the activation function at the given input using the
/// ActivationFunction's `activate` function. We can use this to compare against the
/// ActivationFunction's `derivative` function to make sure it's correct.
///
/// We're using the the centered difference formula for better accuracy: (f(x + h) - f(x - h)) / 2h
/// The normal finite difference formula has less accuracy: (f(x + h) - f(x)) / h
fn estimateSlopeOfActivationFunction(
    activation_function: ActivationFunction,
    inputs: []const f64,
    // Find out how much output of the activate function at the given `input_index`
    // changes if we make a nudge to the input at `input_to_nudge_index`
    input_index: usize,
    input_to_nudge_index: usize,
) !f64 {
    var mutable_inputs = try std.testing.allocator.alloc(f64, inputs.len);
    defer std.testing.allocator.free(mutable_inputs);
    @memcpy(mutable_inputs, inputs);

    // We want h to be small but not too small to cause float point precision problems.
    const h = 0.0001;

    // Make a small nudge to the input in the positive direction (+ h)
    mutable_inputs[input_to_nudge_index] += h;
    // Check how much that nudge causes the result to change
    const result1 = activation_function.activate(mutable_inputs, input_index);

    // Make a small nudge to the weight in the negative direction (- h). We
    // `- 2h` because we nudged the weight in the positive direction by
    // `h` just above and want to get back original_value first so we
    // minus h, and then minus h again to get to (- h).
    mutable_inputs[input_to_nudge_index] -= 2 * h;
    // Check how much that nudge causes the cost to change
    const result2 = activation_function.activate(mutable_inputs, input_index);
    // Find how much the cost changed between the two nudges
    const delta_result = result1 - result2;

    // Reset the input back to its original value
    mutable_inputs[input_to_nudge_index] += h;

    // Calculate the gradient: change in activation / change in input (which is 2h)
    const estimated_slope = delta_result / (2 * h);

    return estimated_slope;
}

const ActivationTestCase = struct {
    activation_function: ActivationFunction,
    inputs: []const f64,
    input_index: usize,
};

// Cross-check the `activate` function against the `derivative` function to make sure they
// relate and match up to each other.
test "Slope check single-input `activation` functions with their derivative" {
    var test_cases = [_]ActivationTestCase{
        .{
            .activation_function = ActivationFunction{ .relu = .{} },
            .inputs = &[_]f64{ 0.1, 0.2, 0.3, 0.4, 0.5 },
            .input_index = 2,
        },
        .{
            .activation_function = ActivationFunction{ .relu = .{} },
            .inputs = &[_]f64{ 0.1, 0.2, 0.3 },
            .input_index = 2,
        },
        // ReLU is not differentiable at 0.0 so our estimatation would run into a kink
        // and make the estimated slope innaccurate. So inaccurate that comparing the
        // derivative function against the estimated slope would fail.
        // .{
        //     .activation_function = ActivationFunction{ .relu = .{} },
        //     .inputs = &[_]f64{ -0.2, 0.1, 0.0, 0.1, 0.2 },
        //     .input_index = 2,
        // },
        .{
            .activation_function = ActivationFunction{ .leaky_relu = .{} },
            .inputs = &[_]f64{ 0.1, 0.2, 0.3, 0.4, 0.5 },
            .input_index = 2,
        },
        .{
            .activation_function = ActivationFunction{ .leaky_relu = .{} },
            .inputs = &[_]f64{ 0.1, 0.2, 0.3 },
            .input_index = 2,
        },
        // LeakyReLU is not differentiable at 0.0 so our estimatation would run into a kink
        // and make the estimated slope innaccurate. So inaccurate that comparing the
        // derivative function against the estimated slope would fail.
        // .{
        //     .activation_function = ActivationFunction{ .leaky_relu = .{} },
        //     .inputs = &[_]f64{ -0.2, 0.1, 0.0, 0.1, 0.2 },
        //     .input_index = 2,
        // },
        .{
            .activation_function = ActivationFunction{ .elu = .{} },
            .inputs = &[_]f64{ 0.1, 0.2, 0.3, 0.4, 0.5 },
            .input_index = 2,
        },
        .{
            .activation_function = ActivationFunction{ .elu = .{} },
            .inputs = &[_]f64{ 0.1, 0.2, 0.3 },
            .input_index = 2,
        },
        .{
            .activation_function = ActivationFunction{ .elu = .{} },
            .inputs = &[_]f64{ -0.2, 0.1, 0.0, 0.1, 0.2 },
            .input_index = 2,
        },
        .{
            .activation_function = ActivationFunction{ .sigmoid = .{} },
            .inputs = &[_]f64{ 0.1, 0.2, 0.3, 0.4, 0.5 },
            .input_index = 2,
        },
        .{
            .activation_function = ActivationFunction{ .sigmoid = .{} },
            .inputs = &[_]f64{ 0.1, 0.2, 0.3 },
            .input_index = 2,
        },
        .{
            .activation_function = ActivationFunction{ .sigmoid = .{} },
            .inputs = &[_]f64{ -0.2, 0.1, 0.0, 0.1, 0.2 },
            .input_index = 2,
        },
    };

    for (test_cases) |test_case| {
        var activation_function = test_case.activation_function;
        var inputs = test_case.inputs;
        const input_index = test_case.input_index;

        // Estimate the slope of the activation function at the given input
        const estimated_slope = try estimateSlopeOfActivationFunction(
            activation_function,
            inputs,
            input_index,
            input_index,
        );

        // A derivative is just the slope of the given function. So the slope returned
        // by the `derivative` function should be the same as the slope we estimated.
        const actual_slope = activation_function.derivative(inputs, input_index);

        // Check to make sure the actual slope is within a certain threshold/tolerance
        // of the estimated slope
        try std.testing.expectApproxEqAbs(estimated_slope, actual_slope, 1e-4);
    }
}

// Cross-check the `activate` function against the `jacobian_row` function to make sure
// they relate and match up to each other.
test "Slope check multi-input `activation` functions with their `jacobian_row`" {
    var test_cases = [_]ActivationTestCase{
        .{
            .activation_function = ActivationFunction{ .soft_max = .{} },
            .inputs = &[_]f64{ 0.1, 0.2, 0.3, 0.4, 0.5 },
            .input_index = 2,
        },
        .{
            .activation_function = ActivationFunction{ .soft_max = .{} },
            .inputs = &[_]f64{ 0.1, 0.2, 0.3 },
            .input_index = 2,
        },
        .{
            .activation_function = ActivationFunction{ .soft_max = .{} },
            .inputs = &[_]f64{ -0.2, 0.1, 0.0, 0.1, 0.2 },
            .input_index = 2,
        },
    };

    for (test_cases) |test_case| {
        var activation_function = test_case.activation_function;
        var inputs = test_case.inputs;
        const row_index = test_case.input_index;

        // A Jacobian matrix allows us to take the derivative a function with respect to
        // all of it's inputs. In this case, we get the partial derivative of the
        // activation function (y_i) with respect to each specific input (x_k).
        //
        // Since a derivative is just the slope of the given function, the slopes
        // returned by the `jacobian_row` function should be the same as the slope we
        // estimate.
        const actual_slopes = try activation_function.jacobian_row(
            inputs,
            test_case.input_index,
            std.testing.allocator,
        );
        defer std.testing.allocator.free(actual_slopes);

        // Loop through each input to find the slope of the activation function with
        // respect to that input
        for (inputs, 0..) |_, input_index| {
            // Estimate the slope of the activation function with the given input (y_i =
            // SoftMax(x_i)) with respect to specific input (x_k). 𝝏y_i/𝝏x_k
            const estimated_slope = try estimateSlopeOfActivationFunction(
                activation_function,
                inputs,
                // We are asking for the slope of y_i
                row_index,
                // with respect to x_k. We nudge x_k and see how much it affects y_i
                input_index,
            );

            // Check to make sure the actual slope is within a certain threshold/tolerance
            // of the estimated slope.
            try std.testing.expectApproxEqAbs(
                estimated_slope,
                actual_slopes[input_index],
                1e-4,
            );
        }
    }
}

pub const ActivationFunction = union(enum) {
    relu: Relu,
    leaky_relu: LeakyRelu,
    elu: ELU,
    sigmoid: Sigmoid,
    soft_max: SoftMax,

    pub fn activate(self: @This(), inputs: []const f64, input_index: usize) f64 {
        return switch (self) {
            inline else => |case| case.activate(inputs, input_index),
        };
    }

    // Returns whether or not the activation function uses a single input to produce a
    // single output. This distinction is useful for optimizations and we can use this
    // to determine whether we need to use the more expensive `jacobian_row` function or
    // an efficient shortcut with the `derivative` function.
    //
    // Hint: Most activation functions are single-input activation functions (except for SoftMax).
    pub fn hasSingleInputActivationFunction(self: @This()) bool {
        return switch (self) {
            inline else => |case| @TypeOf(case).has_single_input_activation_function,
        };
    }

    /// A derivative is just the slope of the activation function at a given point.
    pub fn derivative(self: @This(), inputs: []const f64, input_index: usize) f64 {
        return switch (self) {
            inline else => |case| {
                if (comptime !@TypeOf(case).has_single_input_activation_function) {
                    log.err(
                        "Using the `derivative` on an {s} activation function that " ++
                            "doesn't have a single input is probably a mistake since it " ++
                            "will give inaccurate results if you're trying to use this with backpropagation",
                        .{self.getName()},
                    );
                    @panic(
                        "It's probably a mistake to use the `derivative` function with this " ++
                            "activation function that uses multiple inputs (see message above)",
                    );
                }

                return case.derivative(inputs, input_index);
            },
        };
    }

    // The `jacobian_row` function produces a row vector of derivatives that we can
    // think of as a row in Jacobian matrix with the same square size as the number of
    // `inputs`. Each item in the row is the partial derivative of the activation
    // function (y_i) with respect to the input at the given index (x_k).
    //
    // Jacobian matrix example:
    // ┏  𝝏y_1  𝝏y_1  𝝏y_1  𝝏y_1  ┓
    // ┃  𝝏x_1  𝝏x_2  𝝏x_3  𝝏x_4  ┃
    // ┃                          ┃
    // ┃  𝝏y_2  𝝏y_2  𝝏y_2  𝝏y_2  ┃
    // ┃  𝝏x_1  𝝏x_2  𝝏x_3  𝝏x_4  ┃
    // ┃                          ┃
    // ┃  𝝏y_3  𝝏y_3  𝝏y_3  𝝏y_3  ┃
    // ┃  𝝏x_1  𝝏x_2  𝝏x_3  𝝏x_4  ┃
    // ┃                          ┃
    // ┃  𝝏y_4  𝝏y_4  𝝏y_4  𝝏y_4  ┃
    // ┗  𝝏x_1  𝝏x_2  𝝏x_3  𝝏x_4  ┛
    //
    // Note: Single-input activation functions do not need to define a `jacobian_row`
    // function and should use the shortcut as described below.
    //
    // Single-input activation functions produce a sparse Jacobian matrix where only the
    // diagonal elements are defined. We can use this characteristic to efficiently
    // shortcut and just use a single `derivative` since the other elements end up
    // getting multiplied by 0 and don't contribute to the result anyway.
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
    pub fn jacobian_row(
        self: @This(),
        inputs: []const f64,
        row_index: usize,
        allocator: std.mem.Allocator,
    ) ![]f64 {
        return switch (self) {
            inline else => |case| {
                // Use the `jacobian_row` function if the activation function has one
                if (comptime std.meta.trait.hasFn("jacobian_row")(@TypeOf(case))) {
                    return case.jacobian_row(inputs, row_index, allocator);
                }
                // Otherwise, if it is a single-input activation function, we can
                // provide a default implementation that just puts the derivatives along
                // the diagonal of the Jacobian matrix.
                else if (comptime @TypeOf(case).has_single_input_activation_function) {
                    var results = try allocator.alloc(f64, inputs.len);
                    @memset(results, 0);
                    // Given the `row_index`, we know what row of the Jacobian matrix
                    // we're on; so we can just put the derivative along the diagonal by placing
                    // at the `row_index` of that row.
                    //
                    // Fore example, when `inputs.len = 4` and row_index = 2`
                    // -> [0, 0, derivative, 0]
                    results[row_index] = self.derivative(inputs, row_index);

                    return results;
                } else {
                    log.err(
                        "Activation function ({s}) does not have a `jacobian_row` function and does " ++
                            "not have a sparse Jacobian matrix so we can't provide a default implementation " ++
                            "using the `derivative` function",
                        .{self.getName()},
                    );
                    return error.ActivationFunctionDoesNotHaveJacobianRowFunction;
                }
            },
        };
    }

    pub fn getName(self: @This()) []const u8 {
        return switch (self) {
            .relu => "ReLU",
            .leaky_relu => "LeakyReLU",
            .elu => "ELU",
            .sigmoid => "Sigmoid",
            .soft_max => "SoftMax",
        };
    }
};
