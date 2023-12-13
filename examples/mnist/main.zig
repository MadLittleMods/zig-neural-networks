const std = @import("std");
const neural_networks = @import("zig-neural-networks");
const mnist_data_point_utils = @import("utils/mnist_data_point_utils.zig");
const save_load_utils = @import("utils/save_load_utils.zig");

// Set the logging levels
pub const std_options = struct {
    pub const log_level = .debug;

    pub const log_scope_levels = &[_]std.log.ScopeLevel{
        .{ .scope = .zig_neural_networks, .level = .debug },
    };
};

/// Adjust as necessary. To make the program run faster, you can reduce the number of
/// images to train on and test on. To make the program more accurate, you can increase
/// the number of images to train on.
const NUM_OF_IMAGES_TO_TRAIN_ON = 60000; // (max 60k)
/// The number of test points to use when we do a full cost breakdown after each epoch
const NUM_OF_IMAGES_TO_TEST_ON = 10000; // (max 10k)
/// We only use a small portion of test points when calculating cost and accuracy while
/// going through the mini-batches in each epoch. This is to make the program run faster.
/// The full cost breakdown is done after each epoch.
const NUM_OF_IMAGES_TO_QUICK_TEST_ON = 100; // (max 10k)

const BATCH_SIZE: u32 = 100;
const LEARN_RATE: f64 = 0.05;
const MOMENTUM = 0.9;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer switch (gpa.deinit()) {
        .ok => {},
        .leak => std.log.err("GPA allocator: Memory leak detected", .{}),
    };

    // Argument parsing
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);
    // `zig build run-mnist -- --resume`
    const should_resume = for (args) |arg| {
        if (std.mem.eql(u8, arg, "--resume")) {
            break true;
        }
    } else false;

    // Getting the training/testing data ready
    // =======================================
    //
    const parsed_mnist_data = try mnist_data_point_utils.getMnistDataPoints(allocator, .{
        .num_images_to_train_on = NUM_OF_IMAGES_TO_TRAIN_ON,
        .num_images_to_test_on = NUM_OF_IMAGES_TO_TEST_ON,
    });
    defer parsed_mnist_data.deinit();
    const mnist_data = parsed_mnist_data.value;

    // Neural network
    // =======================================
    //
    var opt_parsed_neural_network: ?std.json.Parsed(neural_networks.NeuralNetwork) = null;
    var neural_network = blk: {
        if (should_resume) {
            const parsed_neural_network = try save_load_utils.loadLatestNeuralNetworkCheckpoint(allocator);
            opt_parsed_neural_network = parsed_neural_network;
            break :blk parsed_neural_network.value;
        } else {
            break :blk try neural_networks.NeuralNetwork.initFromLayerSizes(
                &[_]u32{ 784, 100, @typeInfo(mnist_data_point_utils.DigitLabel).Enum.fields.len },
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
                    // .squared_error = .{},
                    .cross_entropy = .{},
                },
                allocator,
            );
        }
    };
    defer if (opt_parsed_neural_network) |parsed_neural_network| {
        // Since parsing uses an arena allocator internally, we can just rely on their
        // `deinit()` method.
        parsed_neural_network.deinit();
    } else {
        defer neural_network.deinit(allocator);
    };

    try train(
        &neural_network,
        &neural_network,
        mnist_data,
        allocator,
    );
}

/// Runs the training loop so the neural network can learn, and prints out progress
/// updates as it goes.
pub fn train(
    neural_network_for_training: *neural_networks.NeuralNetwork,
    neural_network_for_testing: *neural_networks.NeuralNetwork,
    mnist_data: mnist_data_point_utils.NeuralNetworkData,
    allocator: std.mem.Allocator,
) !void {
    const start_timestamp_seconds = std.time.timestamp();

    var current_epoch_index: usize = 0;
    while (
    // true
    current_epoch_index < 1) : (current_epoch_index += 1) {
        // We assume the data is already shuffled so we skip shuffling on the first
        // epoch. Using a pre-shuffled dataset also gives us nice reproducible results
        // during the first epoch when trying to debug things  (like gradient checking).
        var shuffled_training_data_points = mnist_data.training_data_points;
        if (current_epoch_index > 0) {
            // Shuffle the data after each epoch
            shuffled_training_data_points = try neural_networks.shuffleData(
                mnist_data.training_data_points,
                allocator,
                .{},
            );
        }
        // Skip freeing on the first epoch since we didn't shuffle anything and
        // assumed it was already shuffled.
        defer if (current_epoch_index > 0) {
            allocator.free(shuffled_training_data_points);
        };

        // Split the training data into mini batches so way we can get through learning
        // iterations faster. It does make the learning progress a bit noisy because the
        // cost landscape is a bit different for each batch but it's fast and apparently
        // the noise can even be beneficial in various ways, like for escaping settle
        // points in the cost gradient (ridgelines between two valleys).
        //
        // Instead of "gradient descent" with the full training set where we can take
        // perfect steps downhill, we're using mini batches here (called "stochastic
        // gradient descent") where we take steps that are mostly in the correct
        // direction downhill which is good enough to eventually get us to the minimum.
        var batch_index: u32 = 0;
        while ( //batch_index < shuffled_training_data_points.len / BATCH_SIZE
        batch_index < 1) : (batch_index += 1) {
            const batch_start_index = batch_index * BATCH_SIZE;
            const batch_end_index = batch_start_index + BATCH_SIZE;
            const training_batch = shuffled_training_data_points[batch_start_index..batch_end_index];

            try neural_network_for_training.learn(
                training_batch,
                LEARN_RATE,
                MOMENTUM,
                allocator,
            );

            // Print out a progress update every so often
            if (batch_index % 5 == 0) {
                const current_timestamp_seconds = std.time.timestamp();
                const runtime_duration_seconds = current_timestamp_seconds - start_timestamp_seconds;

                const cost = try neural_network_for_testing.cost_many(
                    mnist_data.testing_data_points[0..NUM_OF_IMAGES_TO_QUICK_TEST_ON],
                    allocator,
                );
                const accuracy = try neural_network_for_testing.getAccuracyAgainstTestingDataPoints(
                    mnist_data.testing_data_points[0..NUM_OF_IMAGES_TO_QUICK_TEST_ON],
                    allocator,
                );
                std.log.debug("epoch {d: <3} batch {d: <3} {s: >12} -> cost {d}, " ++
                    "accuracy with {d} test points {d}", .{
                    current_epoch_index,
                    batch_index,
                    std.fmt.fmtDurationSigned(runtime_duration_seconds * std.time.ns_per_s),
                    cost,
                    NUM_OF_IMAGES_TO_QUICK_TEST_ON,
                    accuracy,
                });
            }
        }

        // Do a full cost break-down with all of the test points after each epoch
        const cost = try neural_network_for_testing.cost_many(mnist_data.testing_data_points, allocator);
        const accuracy = try neural_network_for_testing.getAccuracyAgainstTestingDataPoints(
            mnist_data.testing_data_points,
            allocator,
        );
        std.log.debug("epoch end {d: <3} {s: >18} -> cost {d}, accuracy with *ALL* test points {d}", .{
            current_epoch_index,
            "",
            cost,
            accuracy,
        });

        try save_load_utils.saveNeuralNetworkCheckpoint(
            neural_network_for_testing,
            current_epoch_index,
            allocator,
        );
    }
}
