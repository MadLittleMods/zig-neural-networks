const std = @import("std");
const bigEndianStructDeserializer = @import("big_endian_struct_deserializer.zig").bigEndianStructDeserializer;

pub const MnistLabelFileHeader = extern struct {
    /// The magic number is used to identify the file type (doesn't really matter to us)
    magic_number: u32,
    number_of_labels: u32,
};

pub const MnistImageFileHeader = extern struct {
    /// The magic number is used to identify the file type (doesn't really matter to us).
    magic_number: u32,
    number_of_images: u32,
    number_of_rows: u32,
    number_of_columns: u32,
};

pub const RawImageData = [28 * 28]u8;
pub const NormalizedRawImageData = [28 * 28]f64;

pub const RawImage = struct {
    width: u8 = 28,
    height: u8 = 28,
    pixels: RawImageData,
};
pub const NormalizedImage = struct {
    width: u8 = 28,
    height: u8 = 28,
    pixels: NormalizedRawImageData,
};

pub const Image = union(enum) {
    raw_image: RawImage,
    normalized_image: NormalizedImage,
};

pub const LabelType = u8;
pub const LabeledImage = struct {
    label: LabelType,
    image: Image,
};

pub fn MnistFileData(comptime HeaderType: type, comptime ItemType: type) type {
    return struct {
        header: HeaderType,
        items: []const ItemType,
    };
}

// This method works against the standard MNIST dataset files, which can be downloaded from:
// http://yann.lecun.com/exdb/mnist/
pub fn readMnistFile(
    comptime HeaderType: type,
    comptime ItemType: type,
    file_path: []const u8,
    comptime numberOfItemsFieldName: []const u8,
    number_of_items_to_read: u32,
    allocator: std.mem.Allocator,
) !MnistFileData(HeaderType, ItemType) {
    const file = try std.fs.cwd().openFile(file_path, .{});
    defer file.close();

    // Use a buffered reader for better performance (less syscalls)
    // (https://zig.news/kristoff/how-to-add-buffering-to-a-writer-reader-in-zig-7jd)
    var buffered_reader = std.io.bufferedReader(file.reader());
    var file_reader = buffered_reader.reader();

    const header = try file_reader.readStructBig(HeaderType);

    // Make sure we don't try to allocate more images than there are in the file
    if (number_of_items_to_read > @field(header, numberOfItemsFieldName)) {
        std.log.err("Trying to read more items than there are in the file {} > {}", .{
            number_of_items_to_read,
            @field(header, numberOfItemsFieldName),
        });
        return error.UnableToReadMoreItemsThanInFile;
    }

    const image_data_array = try allocator.alloc(ItemType, number_of_items_to_read);
    const deserializer = bigEndianStructDeserializer(file_reader);
    for (0..number_of_items_to_read) |image_index| {
        const image = try deserializer.read(ItemType);
        image_data_array[image_index] = image;
    }

    return .{
        .header = header,
        .items = image_data_array[0..],
    };
}

const MAX_NUM_MNIST_TRAIN_DATA = 60000;
const MAX_NUM_MNIST_TEST_DATA = 10000;

fn concatCurrentDirectory(comptime suffix: []const u8) []const u8 {
    const root_dir = std.fs.path.dirname(@src().file) orelse ".";
    if (suffix[0] == '/') {
        return root_dir ++ suffix;
    }
    return root_dir ++ "/" ++ suffix;
}
const DEFAULT_TRAIN_DATA_FILE_PATH = concatCurrentDirectory("data/train-images-idx3-ubyte");
const DEFAULT_TRAIN_LABELS_FILE_PATH = concatCurrentDirectory("data/train-labels-idx1-ubyte");
const DEFAULT_TEST_DATA_FILE_PATH = concatCurrentDirectory("data/t10k-images-idx3-ubyte");
const DEFAULT_TEST_LABELS_FILE_PATH = concatCurrentDirectory("data/t10k-labels-idx1-ubyte");

const MnistData = struct {
    training_labels: []const LabelType,
    training_images: []const RawImageData,
    testing_labels: []const LabelType,
    testing_images: []const RawImageData,

    pub fn deinit(self: @This(), allocator: std.mem.Allocator) void {
        allocator.free(self.training_labels);
        allocator.free(self.training_images);
        allocator.free(self.testing_labels);
        allocator.free(self.testing_images);
    }
};

pub fn getMnistData(
    allocator: std.mem.Allocator,
    options: struct {
        num_images_to_train_on: u32 = 60000, // (max 60k)
        num_images_to_test_on: u32 = 10000, // (max 10k)

        train_data_file_path: []const u8 = DEFAULT_TRAIN_DATA_FILE_PATH,
        train_labels_file_path: []const u8 = DEFAULT_TRAIN_LABELS_FILE_PATH,
        test_data_file_path: []const u8 = DEFAULT_TEST_DATA_FILE_PATH,
        test_labels_file_path: []const u8 = DEFAULT_TEST_LABELS_FILE_PATH,
    },
) !MnistData {
    if (options.num_images_to_train_on > MAX_NUM_MNIST_TRAIN_DATA) {
        std.log.err("Trying to read more train images than there are in the file {} > {}", .{
            options.num_images_to_train_on,
            MAX_NUM_MNIST_TRAIN_DATA,
        });
        return error.UnableToReadMoreTrainItemsThanInFile;
    }

    if (options.num_images_to_test_on > MAX_NUM_MNIST_TEST_DATA) {
        std.log.err("Trying to read more test images than there are in the file {} > {}", .{
            options.num_images_to_test_on,
            MAX_NUM_MNIST_TEST_DATA,
        });
        return error.UnableToReadMoreTestItemsThanInFile;
    }

    // Read in the MNIST training labels
    const training_labels_data = try readMnistFile(
        MnistLabelFileHeader,
        LabelType,
        options.train_labels_file_path,
        "number_of_labels",
        options.num_images_to_train_on,
        allocator,
    );
    // std.log.debug("training labels header {}", .{training_labels_data.header});
    try std.testing.expectEqual(training_labels_data.header.magic_number, 2049);
    try std.testing.expectEqual(training_labels_data.header.number_of_labels, 60000);

    // Read in the MNIST training images
    const training_images_data = try readMnistFile(
        MnistImageFileHeader,
        RawImageData,
        options.train_data_file_path,
        "number_of_images",
        options.num_images_to_train_on,
        allocator,
    );
    // std.log.debug("training images header {}", .{training_images_data.header});
    try std.testing.expectEqual(training_images_data.header.magic_number, 2051);
    try std.testing.expectEqual(training_images_data.header.number_of_images, 60000);
    try std.testing.expectEqual(training_images_data.header.number_of_rows, 28);
    try std.testing.expectEqual(training_images_data.header.number_of_columns, 28);

    // Read in the MNIST testing labels
    const testing_labels_data = try readMnistFile(
        MnistLabelFileHeader,
        LabelType,
        options.test_labels_file_path,
        "number_of_labels",
        options.num_images_to_test_on,
        allocator,
    );
    // std.log.debug("testing labels header {}", .{testing_labels_data.header});
    try std.testing.expectEqual(testing_labels_data.header.magic_number, 2049);
    try std.testing.expectEqual(testing_labels_data.header.number_of_labels, 10000);

    // Read in the MNIST testing images
    const testing_images_data = try readMnistFile(
        MnistImageFileHeader,
        RawImageData,
        options.test_data_file_path,
        "number_of_images",
        options.num_images_to_test_on,
        allocator,
    );
    // std.log.debug("testing images header {}", .{testing_images_data.header});
    try std.testing.expectEqual(testing_images_data.header.magic_number, 2051);
    try std.testing.expectEqual(testing_images_data.header.number_of_images, 10000);
    try std.testing.expectEqual(testing_images_data.header.number_of_rows, 28);
    try std.testing.expectEqual(testing_images_data.header.number_of_columns, 28);

    return .{
        .training_labels = training_labels_data.items,
        .training_images = training_images_data.items,
        .testing_labels = testing_labels_data.items,
        .testing_images = testing_images_data.items,
    };
}

pub fn normalizeMnistRawImageData(
    raw_images: []const RawImageData,
    allocator: std.mem.Allocator,
) ![]NormalizedRawImageData {
    const normalized_raw_images = try allocator.alloc(NormalizedRawImageData, raw_images.len);
    for (raw_images, 0..) |raw_image, image_index| {
        for (raw_image, 0..) |pixel, pixel_index| {
            normalized_raw_images[image_index][pixel_index] = @as(f64, @floatFromInt(pixel)) / 255.0;
        }
    }
    return normalized_raw_images[0..];
}
