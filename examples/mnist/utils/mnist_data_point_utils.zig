const std = @import("std");
const neural_networks = @import("zig-neural-networks");
const mnist_data_utils = @import("mnist_data_utils.zig");
const mnist_print_utils = @import("print_utils.zig");

const DataPoint = neural_networks.DataPoint;

pub const DigitLabel = enum(u8) {
    zero = 0,
    one = 1,
    two = 2,
    three = 3,
    four = 4,
    five = 5,
    six = 6,
    seven = 7,
    eight = 8,
    nine = 9,
};
pub const one_hot_digit_label_map = neural_networks.convertLabelEnumToOneHotEncodedEnumMap(DigitLabel);

/// Based on `std.json.Parsed`. Just a good pattern to have everything use an arena
/// allocator and pass a `deinit` function back to free all of the memory.
pub fn Parsed(comptime T: type) type {
    return struct {
        arena: *std.heap.ArenaAllocator,
        value: T,

        pub fn deinit(self: @This()) void {
            const allocator = self.arena.child_allocator;
            self.arena.deinit();
            allocator.destroy(self.arena);
        }
    };
}

pub const NeuralNetworkData = struct {
    training_data_points: []DataPoint,
    testing_data_points: []DataPoint,
};

/// Handles reading the MNIST data from the filesystem and normalizing it so the pixel
/// values are [0-1].
pub fn getMnistDataPoints(
    base_allocator: std.mem.Allocator,
    options: struct {
        num_images_to_train_on: u32 = 60000, // (max 60k)
        num_images_to_test_on: u32 = 10000, // (max 10k)
    },
) !Parsed(NeuralNetworkData) {
    var parsed = Parsed(NeuralNetworkData){
        .arena = try base_allocator.create(std.heap.ArenaAllocator),
        .value = undefined,
    };
    errdefer base_allocator.destroy(parsed.arena);
    parsed.arena.* = std.heap.ArenaAllocator.init(base_allocator);
    errdefer parsed.arena.deinit();

    const allocator = parsed.arena.allocator();

    // Read the MNIST data from the filesystem and normalize it.
    const raw_mnist_data = try mnist_data_utils.getMnistData(allocator, .{
        .num_images_to_train_on = options.num_images_to_train_on,
        .num_images_to_test_on = options.num_images_to_test_on,
    });
    // defer raw_mnist_data.deinit(allocator);
    const normalized_raw_training_images = try mnist_data_utils.normalizeMnistRawImageData(
        raw_mnist_data.training_images,
        allocator,
    );
    // We can't free this yet because our data points reference this memory
    // defer allocator.free(normalized_raw_training_images);
    const normalized_raw_test_images = try mnist_data_utils.normalizeMnistRawImageData(
        raw_mnist_data.testing_images,
        allocator,
    );
    // We can't free this yet because our data points reference this memory
    // defer allocator.free(normalized_raw_test_images);

    // Convert the normalized MNIST data into `DataPoint` which are compatible with the neural network
    var training_data_points = try allocator.alloc(DataPoint, normalized_raw_training_images.len);
    for (normalized_raw_training_images, 0..) |*raw_image, image_index| {
        const label: DigitLabel = @enumFromInt(raw_mnist_data.training_labels[image_index]);
        training_data_points[image_index] = DataPoint.init(
            raw_image,
            // FIXME: Once https://github.com/ziglang/zig/pull/18112 merges and we support a Zig
            // version that includes it, we should use `getPtrConstAssertContains(...)` instead.
            one_hot_digit_label_map.getPtrConst(label).?,
        );
    }
    const testing_data_points = try allocator.alloc(DataPoint, normalized_raw_test_images.len);
    for (normalized_raw_test_images, 0..) |*raw_image, image_index| {
        const label: DigitLabel = @enumFromInt(raw_mnist_data.testing_labels[image_index]);
        testing_data_points[image_index] = DataPoint.init(
            raw_image,
            // FIXME: Once https://github.com/ziglang/zig/pull/18112 merges and we support a Zig
            // version that includes it, we should use `getPtrConstAssertContains(...)` instead.
            one_hot_digit_label_map.getPtrConst(label).?,
        );
    }
    std.log.debug("Created normalized data points. Training on {d} data points, testing on {d}", .{
        training_data_points.len,
        testing_data_points.len,
    });
    // Show what the first image looks like
    std.log.debug("Here is what the first training data point looks like:", .{});
    const expected_label1 = @as(DigitLabel, @enumFromInt(try neural_networks.argmaxOneHotEncodedValue(
        training_data_points[0].expected_outputs,
    )));
    const labeled_image_under_training = mnist_data_utils.LabeledImage{
        .label = @intFromEnum(expected_label1),
        .image = mnist_data_utils.Image{ .normalized_image = .{
            .pixels = training_data_points[0].inputs[0..(28 * 28)].*,
        } },
    };
    try mnist_print_utils.printLabeledImage(labeled_image_under_training, allocator);
    // Sanity check our data, the first training image should be a 5
    try std.testing.expectEqual(
        DigitLabel.five,
        expected_label1,
    );

    parsed.value = NeuralNetworkData{
        .training_data_points = training_data_points,
        .testing_data_points = testing_data_points,
    };

    return parsed;
}
