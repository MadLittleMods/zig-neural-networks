const std = @import("std");
const log = std.log.scoped(.zig_neural_networks);

const DataPoint = @import("../../data_point.zig").DataPoint;
const NeuralNetwork = @import("../../NeuralNetwork.zig");
const Layer = @import("../../layers/Layer.zig");
pub const DenseLayer = @import("../../layers/DenseLayer.zig");

/// Here are the thresholds of error we should be tolerating:
///
/// >  - relative error > 1e-2 usually means the gradient is probably wrong
/// >  - 1e-2 > relative error > 1e-4 should make you feel uncomfortable
/// >  - 1e-4 > relative error is usually okay for objectives with kinks.
/// >    But if there are no kinks (e.g. use of tanh nonlinearities and softmax),
/// >    then 1e-4 is too high.
/// >  - 1e-7 and less you should be happy.
/// >
/// > -- https://cs231n.github.io/neural-networks-3/#gradcheck
fn calculateRelativeError(a: f64, b: f64) f64 {
    // We have this check to watch out for divide by zero
    if (a != 0 and b != 0) {
        // > Notice that normally the relative error formula only includes one of
        // > the two terms (either one), but I prefer to max (or add) both to make
        // > it symmetric and to prevent dividing by zero in the case where one of
        // > the two is zero (which can often happen, especially with ReLUs)
        // >
        // > -- https://cs231n.github.io/neural-networks-3/#gradcheck
        //
        // |a - b| / max(|a|, |b|)
        const relative_error = @fabs(a - b) / @max(@fabs(a), @fabs(b));
        return relative_error;
    }

    return 0;
}

/// Used for gradient checking to calculate the esimated gradient to compare
/// against (also known as the "numerical gradient" as opposed to the actual
/// "analytical gradient").
///
/// This is extremely slow because we have to run the cost function for every weight
/// and bias in the network against all of the training data points.
///
/// Resources:
///  - https://cs231n.github.io/neural-networks-3/#gradcheck
///  - This looks like the naive way (finite-difference) to estimate the slope that
///    Sebastian Lague started off with in his video,
///    https://youtu.be/hfMk-kjRv4c?si=iQohVzk-oFtYldQK&t=937
pub fn estimateCostGradientsForLayer(
    neural_network: *NeuralNetwork,
    layer: *DenseLayer,
    training_data_batch: []const DataPoint,
    allocator: std.mem.Allocator,
) !struct {
    cost_gradient_weights: []f64,
    cost_gradient_biases: []f64,
} {
    var cost_gradient_weights: []f64 = try allocator.alloc(f64, layer.parameters.num_input_nodes * layer.parameters.num_output_nodes);
    var cost_gradient_biases: []f64 = try allocator.alloc(f64, layer.parameters.num_output_nodes);

    // We want h to be small but not too small to cause float point precision problems.
    const h: f64 = 0.0001;

    // Calculate the cost gradient for the current weights.
    // We're using the the centered difference formula for better accuracy: (f(x + h) - f(x - h)) / 2h
    // The normal finite difference formula has less accuracy: (f(x + h) - f(x)) / h
    //
    // We need to be aware of kinks in the objective introduced by activation
    // functions like ReLU. Imagine our weight is just below 0 on the x-axis and
    // we nudge the weight just above 0, we would estimate some value when the
    // actual value is 0 (with ReLU(x), any x <= 0 will result in 0). Ideally,
    // we should have some leniency during the gradient check as it's expected
    // that our estimated gradient will not match our actual gradient exactly
    // when we hit a kink.
    for (0..layer.parameters.num_output_nodes) |node_index| {
        for (0..layer.parameters.num_input_nodes) |node_in_index| {
            const weight_index = layer.getFlatWeightIndex(node_index, node_in_index);

            // Make a small nudge to the weight in the positive direction (+ h)
            layer.parameters.weights[weight_index] += h;
            // Check how much that nudge causes the cost to change
            const cost1 = try neural_network.cost_many(training_data_batch, allocator);

            // Make a small nudge to the weight in the negative direction (- h). We
            // `- 2h` because we nudged the weight in the positive direction by
            // `h` just above and want to get back original_value first so we
            // minus h, and then minus h again to get to (- h).
            layer.parameters.weights[weight_index] -= 2 * h;
            // Check how much that nudge causes the cost to change
            const cost2 = try neural_network.cost_many(training_data_batch, allocator);
            // Find how much the cost changed between the two nudges
            const delta_cost = cost1 - cost2;

            // Reset the weight back to its original value
            layer.parameters.weights[weight_index] += h;

            // Calculate the gradient: change in cost / change in weight (which is 2h)
            cost_gradient_weights[weight_index] = delta_cost / (2 * h);
        }
    }

    // Calculate the cost gradient for the current biases
    for (0..layer.parameters.num_output_nodes) |node_index| {
        // Make a small nudge to the bias (+ h)
        layer.parameters.biases[node_index] += h;
        // Check how much that nudge causes the cost to change
        const cost1 = try neural_network.cost_many(training_data_batch, allocator);

        // Make a small nudge to the bias in the negative direction (- h). We
        // `- 2h` because we nudged the bias in the positive direction by
        // `h` just above and want to get back original_value first so we
        // minus h, and then minus h again to get to (- h).
        layer.parameters.biases[node_index] -= 2 * h;
        // Check how much that nudge causes the cost to change
        const cost2 = try neural_network.cost_many(training_data_batch, allocator);
        // Find how much the cost changed between the two nudges
        const delta_cost = cost1 - cost2;

        // Reset the bias back to its original value
        layer.parameters.biases[node_index] += h;

        // Calculate the gradient: change in cost / change in bias (which is 2h)
        cost_gradient_biases[node_index] = delta_cost / (2 * h);
    }

    return .{
        .cost_gradient_weights = cost_gradient_weights,
        .cost_gradient_biases = cost_gradient_biases,
    };
}

/// Gradient checking to make sure our back propagration algorithm is working correctly.
///
/// Check to make sure cost gradients for weights generated from backpropagation
/// and biases match the estimated cost gradients which are easier to trust (less
/// moving pieces). We check to make sure that the actual and esimated gradients
/// match or are a consistent multiple of each other.
pub fn sanityCheckCostGradients(
    neural_network: *NeuralNetwork,
    layer: *DenseLayer,
    training_data_batch: []const DataPoint,
    allocator: std.mem.Allocator,
) !void {
    // Also known as the "numerical gradient" as opposed to the actual
    // "analytical gradient"
    const estimated_cost_gradients = try estimateCostGradientsForLayer(
        neural_network,
        layer,
        training_data_batch,
        allocator,
    );
    defer allocator.free(estimated_cost_gradients.cost_gradient_weights);
    defer allocator.free(estimated_cost_gradients.cost_gradient_biases);

    const gradients_to_compare = [_]struct { gradient_name: []const u8, actual_gradient: []f64, estimated_gradient: []f64 }{
        .{
            .gradient_name = "weight",
            .actual_gradient = layer.cost_gradient_weights,
            .estimated_gradient = estimated_cost_gradients.cost_gradient_weights,
        },
        .{
            .gradient_name = "bias",
            .actual_gradient = layer.cost_gradient_biases,
            .estimated_gradient = estimated_cost_gradients.cost_gradient_biases,
        },
    };

    var found_relative_error: f64 = 0;
    var has_uneven_cost_gradient: bool = false;
    var was_relative_error_too_high: bool = false;
    for (gradients_to_compare) |gradient_to_compare| {
        // Calculate the relative error between the values in the estimated and
        // actual cost gradients. We want to make sure the relative error is not
        // too high.
        // =========================================================================
        for (gradient_to_compare.actual_gradient, gradient_to_compare.estimated_gradient, 0..) |a_value, b_value, gradient_index| {
            const relative_error = calculateRelativeError(a_value, b_value);
            // Set if it's not already set
            if (found_relative_error == 0) {
                found_relative_error = relative_error;
            }

            // Here are the thresholds of error we should be tolerating:
            //
            // >  - relative error > 1e-2 usually means the gradient is probably wrong
            // >  - 1e-2 > relative error > 1e-4 should make you feel uncomfortable
            // >  - 1e-4 > relative error is usually okay for objectives with kinks.
            // >    But if there are no kinks (e.g. use of tanh nonlinearities and softmax),
            // >    then 1e-4 is too high.
            // >  - 1e-7 and less you should be happy.
            // >
            // > -- https://cs231n.github.io/neural-networks-3/#gradcheck
            if (relative_error > 1e-2) {
                log.err("Relative error for index {d} in {s} gradient was too high ({d}).", .{
                    gradient_index,
                    gradient_to_compare.gradient_name,
                    relative_error,
                });
                was_relative_error_too_high = true;
            } else if (relative_error > 1e-4) {
                // > Note that it is possible to know if a kink was crossed in the
                // > evaluation of the loss. This can be done by keeping track of
                // > the identities of all "winners" in a function of form max(x,y);
                // > That is, was x or y higher during the forward pass. If the
                // > identity of at least one winner changes when evaluating f(x+h)
                // > and then f(x−h), then a kink was crossed and the numerical
                // > gradient will not be exact.
                // >
                // > -- https://cs231n.github.io/neural-networks-3/#gradcheck
                log.warn("Relative error for index {d} in {s} gradient is pretty high " ++
                    "but if there was a kink in the objective, this level of error ({d}) is acceptable when " ++
                    "crossing one of those kinks.", .{
                    gradient_index,
                    gradient_to_compare.gradient_name,
                    relative_error,
                });
            }

            if (
            // Compare the error to the first non-zero error we found. If the
            // error is too different then that's suspect since we would expect
            // the error to be the same for all weights/biases.
            std.math.approxEqAbs(f64, relative_error, found_relative_error, 1e-4) and
                // We can also sanity check whether the error is close to 0
                // since that means the estimated and actual cost gradients are
                // roughly the same.
                relative_error > 1e-4)
            {
                has_uneven_cost_gradient = true;
            }
        }
    }

    if (found_relative_error == 0) {
        log.err("Unable to find relative error because the " ++
            "cost gradients had zeros everywhere (at least there was never a spot " ++
            "where both had a non-zero number). Maybe check for a vanishing gradient " ++
            "problem." ++
            "\n    Estimated weight gradient: {d:.6}" ++
            "\n       Actual weight gradient: {d:.6}" ++
            "\n    Estimated bias gradient: {d:.6}" ++
            "\n       Actual bias gradient: {d:.6}", .{
            estimated_cost_gradients.cost_gradient_weights,
            layer.cost_gradient_weights,
            estimated_cost_gradients.cost_gradient_biases,
            layer.cost_gradient_biases,
        });
        return error.UnableToFindRelativeErrorOfEstimatedToActualGradient;
    } else if (found_relative_error > 1e-4) {
        const uneven_error_message = "The relative error is the different across the entire gradient which " ++
            "means the gradient is pointing in a totally different direction than it should. " ++
            "Our backpropagation algorithm is probably wrong.";

        const even_error_message = "The relative error is the same across the entire gradient so even though " ++
            "the actual value is different than the estimated value, it doesn't affect the direction " ++
            "of the gradient or accuracy of the gradient descent step but may indicate some " ++
            "slight problem.";

        // This is just a warning because I don't think it affects the
        // direction. If that assumption is wrong, then this should be an error.
        log.warn("The first relative error we found is {d} " ++
            "(should be ~0 which indicates the estimated and actual gradients match) " ++
            "which means our actual cost gradient values are some multiple of the estimated weight gradient. " ++
            "{s}" ++
            "\n    Estimated weight gradient: {d:.6}" ++
            "\n       Actual weight gradient: {d:.6}" ++
            "\n    Estimated bias gradient: {d:.6}" ++
            "\n       Actual bias gradient: {d:.6}", .{
            found_relative_error,
            if (has_uneven_cost_gradient) uneven_error_message else even_error_message,
            estimated_cost_gradients.cost_gradient_weights,
            layer.cost_gradient_weights,
            estimated_cost_gradients.cost_gradient_biases,
            layer.cost_gradient_biases,
        });
    }

    // We only consider it an error when the error was too high and the error
    // was inconsisent across the gradient which means we're just stepping in
    // a completely wrong direction.
    if (was_relative_error_too_high and has_uneven_cost_gradient) {
        log.err("Relative error in cost gradients was too high meaning that " ++
            "some values in the estimated vs actual cost gradients were too different " ++
            "which means our backpropagation algorithm is probably wrong and we're " ++
            "probably stepping in an arbitrarily wrong direction. " ++
            "\n    Estimated weight gradient: {d:.6}" ++
            "\n       Actual weight gradient: {d:.6}" ++
            "\n    Estimated bias gradient: {d:.6}" ++
            "\n       Actual bias gradient: {d:.6}", .{
            estimated_cost_gradients.cost_gradient_weights,
            layer.cost_gradient_weights,
            estimated_cost_gradients.cost_gradient_biases,
            layer.cost_gradient_biases,
        });
        return error.RelativeErrorTooHigh;
    }
}
