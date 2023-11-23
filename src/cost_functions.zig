const std = @import("std");
const log = std.log.scoped(.zig_neural_networks);

// Math equation references (loss and derivative):
// https://stats.stackexchange.com/questions/154879/a-list-of-cost-functions-used-in-neural-networks-alongside-applications/154880#154880

// TODO: In the future, we could add negative log likelihood, MeanAbsoluteError (L1 loss),
// RootMeanSquaredError, Focal Loss,  etc.

/// SquaredError (also known as L2 loss)
///
/// Used for regression problems where the output data comes from a normal/gaussian
/// distribution like predicting something based on a trend (extrapolate). Does not
/// penalize misclassifications as much as it could/should for binary/multi-class
/// classification problems. Although this answer says that it doesn't matter,
/// https://stats.stackexchange.com/a/568253/360344. Useful when TODO: What?
pub const SquaredError = struct {
    // Sum of Squared Errors (SSE)
    pub fn vector_cost(
        self: @This(),
        actual_outputs: []const f64,
        expected_outputs: []const f64,
    ) f64 {
        var cost_sum: f64 = 0;
        for (actual_outputs, expected_outputs) |actual_output, expected_output| {
            cost_sum += self.individual_cost(actual_output, expected_output);
        }

        // We want to calculate the total cost (not the average cost).
        return cost_sum;
    }

    pub fn individual_cost(_: @This(), actual_output: f64, expected_output: f64) f64 {
        const error_difference = actual_output - expected_output;
        const squared_error = (error_difference * error_difference);

        // > We multiply our cost function by 1/2 so that when we take the derivative,
        // > the 2s cancel out. Multiplying the cost function by a scalar does not affect
        // > the location of its minimum, so we can get away with this.
        // >
        // > https://mccormickml.com/2014/03/04/gradient-descent-derivation/#one-half-mean-squared-error
        const one_half_squared_error = 0.5 * squared_error;

        return one_half_squared_error;
    }

    pub fn individual_derivative(_: @This(), actual_output: f64, expected_output: f64) f64 {
        return actual_output - expected_output;
    }
};

/// Cross-Entropy is also referred to as Logarithmic loss. Used for binary or multi-class
/// classification problems where the output data comes from a bernoulli distribution
/// which just means we have buckets/categories with expected probabilities.
///
/// Also called Sigmoid Cross-Entropy loss or Binary Cross-Entropy Loss
/// https://gombru.github.io/2018/05/23/cross_entropy_loss/
///
/// Why use Cross-Entropy for loss?
///
///  1. The first reason is that it gives steeper gradients for regimes that matter which
///     results in fewer training epochs to get to the local minimum.
///  2. The second and probably more important reason is that when you're predicting more
///     than one class (e.g., not just "cat" or "not", but also a third like "cat",
///     "dog", or "neither"), Cross-Entropy gives you calibrated[1] results, whereas
///     SquaredError will not necessarily.
///     - [1] To explain what "calibrated" means; say you want a composite model. The
///       first predicts the probability that a customer will convert in one of 3
///       different price buckets. The second then multiplies the probability by the size
///       of the bucket ($5, $30, or $80 let's say). If there's any reason the model is
///       more accurate for some classes than others (extremely common), then an error in
///       the $5 bucket has very different effects on the resulting estimate of lifetime
///       customer value than an error in the $80 bucket. If your probabilities are
///       calibrated then you can blindly do the aforementioned multiplication and know
///       that on average it's correct. Otherwise, you're prone to (drastically) under or
///       over valuing a customer and making incorrect decisions as a result of that
///       information.
///
/// With just 2 classes, the optimal values of SquaredError and Cross-Entropy are
/// identical, so only learning rate applies. With 3 or more, Cross-Entropy is
/// potentially more calibrated any time there is shared information in the inputs. The
/// simplest case of that is when two inputs overlap completely, but a neural network
/// maps close inputs to similar outputs (caveats apply), so for vector-valued inputs
/// like images (where you feed in all of the pixels) you'll see the same effect just
/// from images that look close to each other.
///
/// https://machinelearningmastery.com/cross-entropy-for-machine-learning/
pub const CrossEntropy = struct {
    pub fn vector_cost(
        self: @This(),
        actual_outputs: []const f64,
        /// Note: `expected_outputs` are expected to all be either 0 or 1 (probably using one-hot encoding).
        expected_outputs: []const f64,
    ) f64 {
        var cost_sum: f64 = 0;
        for (actual_outputs, expected_outputs) |actual_output, expected_output| {
            cost_sum += self.individual_cost(actual_output, expected_output);
        }

        // We want to calculate the total cost (not the average cost).
        return cost_sum;
    }

    // Reference breakdown of the equations: https://www.youtube.com/watch?v=DPSXVJF5jIs
    // (Understanding Binary Cross-Entropy / Log Loss in 5 minutes: a visual explanation) by Daniel Godoy
    // (or associated article: https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a)
    pub fn individual_cost(
        _: @This(),
        actual_output: f64,
        /// Note: `expected_output` should be either 0.0 or 1.0.
        expected_output: f64,
    ) f64 {
        var v: f64 = 0;
        if (expected_output == 1.0) {
            v += -1 * @log(actual_output);
        } else if (expected_output == 0.0) {
            v += -1 * @log(1 - actual_output);
        } else {
            log.err("CrossEntropy.individual_cost(): Expected `expected_output` to be either 0.0 or 1.0 but was given {d}", .{expected_output});
            @panic("CrossEntropy.individual_cost(): Expected `expected_output` to be either 0.0 or 1.0");
        }

        return if (std.math.isNan(v)) 0 else v;
    }

    pub fn individual_derivative(_: @This(), actual_output: f64, expected_output: f64) f64 {
        // The function is undefined at 0 and 1 (not continuous/differentiable), so we
        // return 0 because TODO: Why?
        if (actual_output == 0.0 or actual_output == 1.0) {
            return 0.0;
        }

        // Alternative form: (actual_output - expected_output) / (actual_output * (1 - actual_output))
        return (expected_output - actual_output) / (actual_output * (actual_output - 1));
    }
};

/// Estimate the slope of the cost function at the given input using the
/// CostFunction's `activate` function. We can use this to compare against the
/// CostFunction's `derivative` function to make sure it's correct.
///
/// We're using the the centered difference formula for better accuracy: (f(x + h) - f(x - h)) / 2h
/// The normal finite difference formula has less accuracy: (f(x + h) - f(x)) / h
fn estimateSlopeOfCostFunction(
    cost_function: CostFunction,
    actual_output: f64,
    expected_output: f64,
) !f64 {
    // We want h to be small but not too small to cause float point precision problems.
    const h = 0.0001;

    var mutable_actual_output = actual_output;

    // Make a small nudge to the input in the positive direction (+ h)
    mutable_actual_output += h;
    // Check how much that nudge causes the result to change
    const result1 = cost_function.individual_cost(mutable_actual_output, expected_output);

    // Make a small nudge to the weight in the negative direction (- h). We
    // `- 2h` because we nudged the weight in the positive direction by
    // `h` just above and want to get back original_value first so we
    // minus h, and then minus h again to get to (- h).
    mutable_actual_output -= 2 * h;
    // Check how much that nudge causes the cost to change
    const result2 = cost_function.individual_cost(mutable_actual_output, expected_output);
    // Find how much the cost changed between the two nudges
    const delta_result = result1 - result2;

    // Reset the input back to its original value
    mutable_actual_output += h;

    // Calculate the gradient: change in activation / change in input (which is 2h)
    const estimated_cost = delta_result / (2 * h);

    return estimated_cost;
}

const LossTestCase = struct {
    cost_function: CostFunction,
    actual_output: f64,
    expected_output: f64,
};

// Cross-check the `individual_cost` function against the `individual_derivative`
// function to make sure they relate and match up to each other.
test "Slope check cost functions" {
    const test_cases = [_]LossTestCase{
        // SquaredError
        .{
            .cost_function = CostFunction{ .squared_error = .{} },
            .actual_output = 0.5,
            .expected_output = 0.75,
        },
        .{
            .cost_function = CostFunction{ .squared_error = .{} },
            .actual_output = 0.75,
            .expected_output = 0.5,
        },
        .{
            .cost_function = CostFunction{ .squared_error = .{} },
            .actual_output = 0.5,
            .expected_output = 0.5,
        },
        .{
            .cost_function = CostFunction{ .squared_error = .{} },
            .actual_output = 0.75,
            .expected_output = 1.0,
        },
        .{
            .cost_function = CostFunction{ .squared_error = .{} },
            .actual_output = -5,
            .expected_output = 0,
        },
        .{
            .cost_function = CostFunction{ .squared_error = .{} },
            .actual_output = 5,
            .expected_output = 0,
        },
        .{
            .cost_function = CostFunction{ .squared_error = .{} },
            .actual_output = 0,
            .expected_output = 0,
        },
        .{
            .cost_function = CostFunction{ .squared_error = .{} },
            .actual_output = 0,
            .expected_output = -5,
        },
        .{
            .cost_function = CostFunction{ .squared_error = .{} },
            .actual_output = 0,
            .expected_output = 5,
        },
        .{
            .cost_function = CostFunction{ .squared_error = .{} },
            .actual_output = 3,
            .expected_output = 1.0,
        },
        // CrossEntropy has some specific preconditions (which is why we use it
        // alongside SoftMax activation function):
        //  - `actual_output` range has to be within (0, 1)
        //  - `expected_output` has to be either 0.0 or 1.0
        .{
            .cost_function = CostFunction{ .cross_entropy = .{} },
            .actual_output = 0.1,
            .expected_output = 1.0,
        },
        .{
            .cost_function = CostFunction{ .cross_entropy = .{} },
            .actual_output = 0.5,
            .expected_output = 1.0,
        },
        .{
            .cost_function = CostFunction{ .cross_entropy = .{} },
            .actual_output = 0.5,
            .expected_output = 0.0,
        },
        .{
            .cost_function = CostFunction{ .cross_entropy = .{} },
            .actual_output = 0.9,
            .expected_output = 0.0,
        },
    };

    for (test_cases) |test_case| {
        var cost_function = test_case.cost_function;
        const actual_output = test_case.actual_output;
        const expected_output = test_case.expected_output;

        // Estimate the slope of the activation function at the given input
        const estimated_slope = try estimateSlopeOfCostFunction(
            cost_function,
            actual_output,
            expected_output,
        );
        // A derivative is just the slope of the given function. So the slope returned
        // by the derivative function should be the same as the slope we estimated.
        const actual_slope = cost_function.individual_derivative(actual_output, expected_output);

        // Check to make sure the actual slope is within a certain threshold/tolerance
        // of the estimated slope
        try std.testing.expectApproxEqAbs(estimated_slope, actual_slope, 1e-4);
    }
}

/// Cost functions are also known as loss/error functions.
pub const CostFunction = union(enum) {
    squared_error: SquaredError,
    cross_entropy: CrossEntropy,

    /// Given the actual output vector and the expected output vector, calculate the cost.
    /// This function returns the total cost (not the average cost).
    pub fn vector_cost(self: @This(), actual_outputs: []const f64, expected_outputs: []const f64) f64 {
        return switch (self) {
            inline else => |case| case.vector_cost(actual_outputs, expected_outputs),
        };
    }

    pub fn individual_cost(self: @This(), actual_output: f64, expected_output: f64) f64 {
        return switch (self) {
            inline else => |case| case.individual_cost(actual_output, expected_output),
        };
    }

    // TODO: Derivative of what with respect to what?
    pub fn individual_derivative(self: @This(), actual_output: f64, expected_output: f64) f64 {
        return switch (self) {
            inline else => |case| case.individual_derivative(actual_output, expected_output),
        };
    }

    pub fn getName(self: @This()) []const u8 {
        return switch (self) {
            .squared_error => "SquaredError",
            .cross_entropy => "CrossEntropy",
        };
    }
};
