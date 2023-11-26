const std = @import("std");
const neural_networks = @import("zig-neural-networks");

// Set the logging levels
pub const std_options = struct {
    pub const log_level = .debug;

    pub const log_scope_levels = &[_]std.log.ScopeLevel{
        .{ .scope = .zig_neural_networks, .level = .debug },
    };
};

const BATCH_SIZE: u32 = 10;
const LEARN_RATE: f64 = 0.1;
// Since this problem space doesn't have much curvature, momentum tends to hurt us more
// with higher values.
const MOMENTUM = 0.3;

// This is a small testing dataset (to make sure our code is working) with only 2
// arbitrary features (x and y) where the labeled data points (fish and goat) occupy
// distinct parts of the graph. The boundary between the two labels is not a
// straight line (linear relationship) so we need the power of a neural network to
// learn the non-linear relationship.
//
// Since we only have two inputs with this dataset, we could graph the data points
// based on the inputs as (x, y) and colored based on the label. Then we can run the
// neural network over every pixel in the graph to visualize the boundary that the
// networks weights and biases is making. See https://youtu.be/hfMk-kjRv4c?t=311 for
// reference.
const DataPoint = neural_networks.DataPoint;
const AnimalLabel = enum {
    fish,
    goat,
};
// This can be a `const` once https://github.com/ziglang/zig/pull/18112 merges and we
// support a Zig version that includes it.
const one_hot_animal_label_map = neural_networks.convertLabelEnumToOneHotEncodedEnumMap(AnimalLabel);
// Graph of animal data points:
// https://www.desmos.com/calculator/tkfacez5wt
var animal_training_data_points = [_]DataPoint{
    // FIXME: Once https://github.com/ziglang/zig/pull/18112 merges and we support a Zig
    // version that includes it, we should use `getPtrConstAssertContains(...)` instead.
    DataPoint.init(&[_]f64{ 0.924, 0.166 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.04, 0.085 }, one_hot_animal_label_map.getPtrConst(.fish).?),
    DataPoint.init(&[_]f64{ 0.352, 0.373 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.662, 0.737 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.724, 0.049 }, one_hot_animal_label_map.getPtrConst(.fish).?),
    DataPoint.init(&[_]f64{ 0.123, 0.517 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.2245, 0.661 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.466, 0.504 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.375, 0.316 }, one_hot_animal_label_map.getPtrConst(.fish).?),
    DataPoint.init(&[_]f64{ 0.039, 0.3475 }, one_hot_animal_label_map.getPtrConst(.fish).?),
    DataPoint.init(&[_]f64{ 0.28, 0.363 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.342, 0.142 }, one_hot_animal_label_map.getPtrConst(.fish).?),
    DataPoint.init(&[_]f64{ 0.517, 0.416 }, one_hot_animal_label_map.getPtrConst(.fish).?),
    DataPoint.init(&[_]f64{ 0.108, 0.403 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.728, 0.208 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.214, 0.238 }, one_hot_animal_label_map.getPtrConst(.fish).?),
    DataPoint.init(&[_]f64{ 0.865, 0.525 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.645, 0.363 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.436, 0.182 }, one_hot_animal_label_map.getPtrConst(.fish).?),
    DataPoint.init(&[_]f64{ 0.41, 0.085 }, one_hot_animal_label_map.getPtrConst(.fish).?),
    DataPoint.init(&[_]f64{ 0.146, 0.404 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.09, 0.457 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.663, 0.61 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.445, 0.384 }, one_hot_animal_label_map.getPtrConst(.fish).?),
    DataPoint.init(&[_]f64{ 0.588, 0.409 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.49, 0.075 }, one_hot_animal_label_map.getPtrConst(.fish).?),
    DataPoint.init(&[_]f64{ 0.679, 0.4455 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.145, 0.159 }, one_hot_animal_label_map.getPtrConst(.fish).?),
    DataPoint.init(&[_]f64{ 0.086, 0.155 }, one_hot_animal_label_map.getPtrConst(.fish).?),
    DataPoint.init(&[_]f64{ 0.192, 0.348 }, one_hot_animal_label_map.getPtrConst(.fish).?),
    DataPoint.init(&[_]f64{ 0.766, 0.62 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.132, 0.28 }, one_hot_animal_label_map.getPtrConst(.fish).?),
    DataPoint.init(&[_]f64{ 0.04, 0.403 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.588, 0.353 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.59, 0.452 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.364, 0.042 }, one_hot_animal_label_map.getPtrConst(.fish).?),
    DataPoint.init(&[_]f64{ 0.863, 0.068 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.806, 0.274 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.571, 0.49 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.762, 0.39 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.245, 0.388 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.097, 0.05 }, one_hot_animal_label_map.getPtrConst(.fish).?),
    DataPoint.init(&[_]f64{ 0.112, 0.339 }, one_hot_animal_label_map.getPtrConst(.fish).?),
    DataPoint.init(&[_]f64{ 0.538, 0.51 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.73, 0.507 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.472, 0.604 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.368, 0.506 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.768, 0.14 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.49, 0.75 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.21, 0.573 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.881, 0.382 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.331, 0.263 }, one_hot_animal_label_map.getPtrConst(.fish).?),
    DataPoint.init(&[_]f64{ 0.6515, 0.213 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.155, 0.721 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.89, 0.746 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.613, 0.265 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.442, 0.449 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.064, 0.554 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.314, 0.771 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.673, 0.135 }, one_hot_animal_label_map.getPtrConst(.fish).?),
    DataPoint.init(&[_]f64{ 0.535, 0.216 }, one_hot_animal_label_map.getPtrConst(.fish).?),
    DataPoint.init(&[_]f64{ 0.047, 0.267 }, one_hot_animal_label_map.getPtrConst(.fish).?),
    DataPoint.init(&[_]f64{ 0.502, 0.324 }, one_hot_animal_label_map.getPtrConst(.fish).?),
    DataPoint.init(&[_]f64{ 0.096, 0.827 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.586, 0.653 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.214, 0.049 }, one_hot_animal_label_map.getPtrConst(.fish).?),
    DataPoint.init(&[_]f64{ 0.683, 0.88 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.246, 0.315 }, one_hot_animal_label_map.getPtrConst(.fish).?),
    DataPoint.init(&[_]f64{ 0.264, 0.512 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.39, 0.414 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.323, 0.573 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.593, 0.307 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.314, 0.692 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.817, 0.456 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.596, 0.054 }, one_hot_animal_label_map.getPtrConst(.fish).?),
    DataPoint.init(&[_]f64{ 0.192, 0.403 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.195, 0.469 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.587, 0.138 }, one_hot_animal_label_map.getPtrConst(.fish).?),
    DataPoint.init(&[_]f64{ 0.315, 0.338 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.917, 0.267 }, one_hot_animal_label_map.getPtrConst(.goat).?),
};
const animal_testing_data_points = [_]DataPoint{
    DataPoint.init(&[_]f64{ 0.23, 0.14 }, one_hot_animal_label_map.getPtrConst(.fish).?),
    DataPoint.init(&[_]f64{ 0.087, 0.236 }, one_hot_animal_label_map.getPtrConst(.fish).?),
    DataPoint.init(&[_]f64{ 0.507, 0.142 }, one_hot_animal_label_map.getPtrConst(.fish).?),
    DataPoint.init(&[_]f64{ 0.503, 0.403 }, one_hot_animal_label_map.getPtrConst(.fish).?),
    DataPoint.init(&[_]f64{ 0.67, 0.076 }, one_hot_animal_label_map.getPtrConst(.fish).?),
    DataPoint.init(&[_]f64{ 0.074, 0.34 }, one_hot_animal_label_map.getPtrConst(.fish).?),
    DataPoint.init(&[_]f64{ 0.41, 0.257 }, one_hot_animal_label_map.getPtrConst(.fish).?),
    DataPoint.init(&[_]f64{ 0.278, 0.273 }, one_hot_animal_label_map.getPtrConst(.fish).?),
    DataPoint.init(&[_]f64{ 0.5065, 0.373 }, one_hot_animal_label_map.getPtrConst(.fish).?),
    DataPoint.init(&[_]f64{ 0.5065, 0.272 }, one_hot_animal_label_map.getPtrConst(.fish).?),
    DataPoint.init(&[_]f64{ 0.551, 0.173 }, one_hot_animal_label_map.getPtrConst(.fish).?),
    DataPoint.init(&[_]f64{ 0.636, 0.128 }, one_hot_animal_label_map.getPtrConst(.fish).?),
    DataPoint.init(&[_]f64{ 0.2, 0.33 }, one_hot_animal_label_map.getPtrConst(.fish).?),
    DataPoint.init(&[_]f64{ 0.409, 0.345 }, one_hot_animal_label_map.getPtrConst(.fish).?),
    DataPoint.init(&[_]f64{ 0.358, 0.284 }, one_hot_animal_label_map.getPtrConst(.fish).?),
    DataPoint.init(&[_]f64{ 0.098, 0.102 }, one_hot_animal_label_map.getPtrConst(.fish).?),
    DataPoint.init(&[_]f64{ 0.442, 0.058 }, one_hot_animal_label_map.getPtrConst(.fish).?),
    DataPoint.init(&[_]f64{ 0.368, 0.167 }, one_hot_animal_label_map.getPtrConst(.fish).?),
    DataPoint.init(&[_]f64{ 0.459, 0.3235 }, one_hot_animal_label_map.getPtrConst(.fish).?),
    DataPoint.init(&[_]f64{ 0.37, 0.674 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.32, 0.43 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.066, 0.628 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.635, 0.527 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.704, 0.305 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.82, 0.137 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.862, 0.305 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.709, 0.679 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.18, 0.527 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.072, 0.405 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.218, 0.408 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.303, 0.357 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.425, 0.443 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.554, 0.505 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.659, 0.251 }, one_hot_animal_label_map.getPtrConst(.goat).?),
    DataPoint.init(&[_]f64{ 0.597, 0.386 }, one_hot_animal_label_map.getPtrConst(.goat).?),
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer {
        switch (gpa.deinit()) {
            .ok => {},
            .leak => std.log.err("GPA allocator: Memory leak detected", .{}),
        }
    }

    const start_timestamp_seconds = std.time.timestamp();

    var neural_network = try neural_networks.NeuralNetwork.initFromLayerSizes(
        &[_]u32{ 2, 10, 10, @typeInfo(AnimalLabel).Enum.fields.len },
        neural_networks.ActivationFunction{
            // .relu = .{},
            // .leaky_relu = .{},
            .elu = .{},
            //.sigmoid = .{},
        },
        neural_networks.ActivationFunction{
            .soft_max = .{},
            // .sigmoid = .{},
        },
        neural_networks.CostFunction{
            //.squared_error = .{},
            .cross_entropy = .{},
        },
        allocator,
    );
    defer neural_network.deinitFromLayerSizes(allocator);

    var current_epoch_index: usize = 0;
    while (true) : (current_epoch_index += 1) {
        // We assume the data is already shuffled so we skip shuffling on the first
        // epoch. Using a pre-shuffled dataset also gives us nice reproducible results
        // during the first epoch when trying to debug things  (like gradient checking).
        var shuffled_training_data_points: []DataPoint = &animal_training_data_points;
        if (current_epoch_index > 0) {
            // Shuffle the data after each epoch
            shuffled_training_data_points = try neural_networks.shuffleData(
                &animal_training_data_points,
                allocator,
                .{},
            );
        }
        defer {
            if (current_epoch_index > 0) {
                allocator.free(shuffled_training_data_points);
            }
        }

        // Split the training data into mini batches so way we can get through learning
        // iterations faster. It does make the learning progress a bit noisy because the
        // cost landscape is a bit different for each batch but it's fast and apparently
        // the noise can even be beneficial in various ways, like for escaping settle
        // points in the cost gradient (ridgelines between two valleys).
        //
        // Instead of "gradient descent" with the full training set, using mini batches
        // is called "stochastic gradient descent".
        var batch_index: u32 = 0;
        while (batch_index < shuffled_training_data_points.len / BATCH_SIZE) : (batch_index += 1) {
            const batch_start_index = batch_index * BATCH_SIZE;
            const batch_end_index = batch_start_index + BATCH_SIZE;
            const training_batch = shuffled_training_data_points[batch_start_index..batch_end_index];

            try neural_network.learn(
                training_batch,
                // TODO: Implement learn rate decay so we take more refined steps the
                // longer we train for.
                LEARN_RATE,
                MOMENTUM,
                allocator,
            );

            if (current_epoch_index % 10 == 0 and
                current_epoch_index != 0 and
                batch_index == 0)
            {
                const current_timestamp_seconds = std.time.timestamp();
                const runtime_duration_seconds = current_timestamp_seconds - start_timestamp_seconds;

                const cost = try neural_network.cost_many(&animal_testing_data_points, allocator);
                const accuracy = try neural_network.getAccuracyAgainstTestingDataPoints(
                    &animal_testing_data_points,
                    allocator,
                );
                std.log.debug("epoch {d: <5} batch {d: <2} {s: >12} -> cost {d}, accuracy with testing points {d}", .{
                    current_epoch_index,
                    batch_index,
                    std.fmt.fmtDurationSigned(runtime_duration_seconds * std.time.ns_per_s),
                    cost,
                    accuracy,
                });
            }
        }

        // Graph how the neural network is learning over time.
        if (current_epoch_index % 1000 == 0 and current_epoch_index != 0) {
            try neural_networks.graphNeuralNetwork(
                "simple_xy_animal_graph.ppm",
                &neural_network,
                &animal_training_data_points,
                &animal_testing_data_points,
                allocator,
            );
        }
    }

    // Graph how the neural network looks at the end of training.
    try neural_networks.graphNeuralNetwork(
        "simple_xy_animal_graph.ppm",
        &neural_network,
        &animal_training_data_points,
        &animal_testing_data_points,
        allocator,
    );
}
