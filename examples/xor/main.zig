const std = @import("std");
const neural_networks = @import("zig-neural-networks");

// Set the logging levels
pub const std_options = struct {
    pub const log_level = .debug;

    pub const log_scope_levels = &[_]std.log.ScopeLevel{
        .{ .scope = .zig_neural_networks, .level = .debug },
    };
};

const BATCH_SIZE: u32 = 4;
const LEARN_RATE: f64 = 0.1;
// Since this problem space doesn't have much curvature, momentum tends to hurt us more
// with higher values.
const MOMENTUM = 0.3;

// Binary value can only be 0 or 1
const xor_labels = [_]u8{
    0,
    1,
};
const XorDataPoint = neural_networks.DataPoint(u8, &xor_labels);
// The XOR data points
var xor_data_points = [_]XorDataPoint{
    XorDataPoint.init(&[_]f64{ 0, 0 }, 0),
    XorDataPoint.init(&[_]f64{ 0, 1 }, 1),
    XorDataPoint.init(&[_]f64{ 1, 0 }, 1),
    XorDataPoint.init(&[_]f64{ 1, 1 }, 0),
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

    var neural_network = try neural_networks.NeuralNetwork(XorDataPoint).initFromLayerSizes(
        &[_]u32{ 2, 3, xor_labels.len },
        neural_networks.ActivationFunction{
            // .relu = .{},
            // .leaky_relu = .{},
            .elu = .{},
            // .sigmoid = .{},
        },
        neural_networks.ActivationFunction{
            .soft_max = .{},
            // .sigmoid = .{},
        },
        neural_networks.CostFunction{
            .squared_error = .{},
            // .cross_entropy = .{},
        },
        allocator,
    );
    defer neural_network.deinitFromLayerSizes(allocator);

    var current_epoch_index: usize = 0;
    while (true) : (current_epoch_index += 1) {
        // We assume the data is already shuffled so we skip shuffling on the first
        // epoch. Using a pre-shuffled dataset also gives us nice reproducible results
        // during the first epoch when trying to debug things (like gradient checking).
        var shuffled_training_data_points: []XorDataPoint = &xor_data_points;
        if (current_epoch_index > 0) {
            // Shuffle the data after each epoch
            shuffled_training_data_points = try neural_networks.shuffleData(
                &xor_data_points,
                allocator,
                .{},
            );
        }
        defer {
            // Skip freeing on the first epoch since we didn't shuffle anything and
            // assumed it was already shuffled.
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

                const cost = try neural_network.cost_many(&xor_data_points, allocator);
                const accuracy = try neural_network.getAccuracyAgainstTestingDataPoints(
                    &xor_data_points,
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
    }
}
