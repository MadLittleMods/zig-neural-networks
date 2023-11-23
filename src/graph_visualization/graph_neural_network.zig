const std = @import("std");
const tracy = @import("../tracy.zig");
const neural_networks = @import("../main.zig");
const createPortablePixMap = @import("create_portable_pix_map.zig").createPortablePixMap;

const ColorPair = struct {
    background_color: u24,
    primary_color: u24,
    secondary_color: u24,
};

const color_pair_map = [_]ColorPair{
    .{
        // Blue
        .background_color = 0x4444aa,
        // Lighter blue
        .primary_color = 0x6666ff,
        // Purple
        .secondary_color = 0xcc33cc,
    },
    .{
        // Red
        .background_color = 0xaa4444,
        // Lighter red
        .primary_color = 0xff6666,
        // Yellow
        .secondary_color = 0xcccc33,
    },
    .{
        // Brown
        .background_color = 0x8f7838,
        // Brown-ish
        .primary_color = 0xae9144,
        // Cyan
        .secondary_color = 0x66b6d1,
    },
};

/// Create a graph of the neural network's decision boundary (we can only visualize
/// this because there are only 2 inputs to the neural network which we can map to
/// the 2d image).
pub fn graphNeuralNetwork(
    comptime file_name: []const u8,
    comptime DataPointType: type,
    neural_network: *neural_networks.NeuralNetwork(DataPointType),
    training_data_points: []const DataPointType,
    test_data_points: []const DataPointType,
    allocator: std.mem.Allocator,
) !void {
    const trace = tracy.trace(@src());
    defer trace.end();
    const width: u32 = 400;
    const height: u32 = 400;
    var pixels: []u24 = try allocator.alloc(u24, width * height);
    defer allocator.free(pixels);

    // For every pixel in the graph, run the neural network and color the pixel based on
    // the output of the neural network.
    for (0..height) |height_index| {
        for (0..width) |width_index| {
            // Normalize the pixel coordinates to be between 0 and 1
            const x = @as(f64, @floatFromInt(width_index)) / @as(f64, @floatFromInt(width));
            const y = @as(f64, @floatFromInt(
                // Flip the Y axis so that the origin (0, 0) in our graph is at the bottom left of the image
                height - height_index - 1,
            )) / @as(f64, @floatFromInt(height));
            const predicted_label = try neural_network.classify(&[_]f64{ x, y }, allocator);

            const predicted_label_index: usize = try DataPointType.labelToOneHotIndex(predicted_label);
            if (predicted_label_index > color_pair_map.len - 1) {
                return error.ColorPairMapNotLargeEnough;
            }

            const pixel_color: u24 = color_pair_map[predicted_label_index].background_color;
            pixels[height_index * width + width_index] = pixel_color;
        }
    }

    // Draw a ball for every training point
    for (training_data_points) |*data_point| {
        const label_index: usize = try DataPointType.labelToOneHotIndex(data_point.label);
        if (label_index > color_pair_map.len - 1) {
            return error.ColorPairMapNotLargeEnough;
        }
        const pixel_color: u24 = color_pair_map[label_index].primary_color;

        // Draw the border/shadow of the ball
        drawBallOnPixelCanvasForDataPoint(
            DataPointType,
            .{
                .pixels = pixels,
                .width = width,
                .height = height,
            },
            data_point,
            10,
            0x111111,
        );

        // Draw the colored part of the ball
        drawBallOnPixelCanvasForDataPoint(
            DataPointType,
            .{
                .pixels = pixels,
                .width = width,
                .height = height,
            },
            data_point,
            8,
            pixel_color,
        );
    }

    // Draw a ball for every test point
    for (test_data_points) |*data_point| {
        const label_index: usize = try DataPointType.labelToOneHotIndex(data_point.label);
        if (label_index > color_pair_map.len - 1) {
            return error.ColorPairMapNotLargeEnough;
        }
        const pixel_color: u24 = color_pair_map[label_index].secondary_color;

        // Draw the border/shadow of the ball
        drawBallOnPixelCanvasForDataPoint(
            DataPointType,
            .{
                .pixels = pixels,
                .width = width,
                .height = height,
            },
            data_point,
            8,
            0x111111,
        );

        // Draw the colored part of the ball
        drawBallOnPixelCanvasForDataPoint(
            DataPointType,
            .{
                .pixels = pixels,
                .width = width,
                .height = height,
            },
            data_point,
            6,
            pixel_color,
        );
    }

    const ppm_file_contents = try createPortablePixMap(pixels, width, height, allocator);
    defer allocator.free(ppm_file_contents);
    const file = try std.fs.cwd().createFile(file_name, .{});
    defer file.close();

    try file.writeAll(ppm_file_contents);
}

fn drawBallOnPixelCanvasForDataPoint(
    comptime DataPointType: type,
    pixel_canvas: struct {
        pixels: []u24,
        width: u32,
        height: u32,
    },
    data_point: *const DataPointType,
    ball_size: u32,
    draw_color: u24,
) void {
    const x_continuous = data_point.inputs[0] * @as(f64, @floatFromInt(pixel_canvas.width));
    const y_continuous = (
    // Flip the Y axis so that the origin (0, 0) in our graph is at the bottom left of the image
        1 - data_point.inputs[1]) * @as(f64, @floatFromInt(pixel_canvas.height));
    const x: i32 = @intFromFloat(x_continuous);
    const y: i32 = @intFromFloat(y_continuous);

    const signed_ball_size: i32 = @as(i32, @intCast(ball_size));
    const ball_start_x: i32 = x - @divTrunc(signed_ball_size, 2);
    const ball_start_y: i32 = y - @divTrunc(signed_ball_size, 2);

    var ball_x_index = ball_start_x;
    while (ball_x_index < ball_start_x + signed_ball_size) : (ball_x_index += 1) {
        var ball_y_index = ball_start_y;
        while (ball_y_index < ball_start_y + signed_ball_size) : (ball_y_index += 1) {
            // Skip any pixels that are outside of the canvas
            if (ball_x_index < 0 or
                ball_x_index >= pixel_canvas.width or
                ball_y_index < 0 or
                ball_y_index >= pixel_canvas.height)
            {
                continue;
            }

            const signed_pixel_index = ball_y_index * @as(i32, @intCast(pixel_canvas.width)) + ball_x_index;
            const pixel_index = @as(u32, @intCast(signed_pixel_index));
            pixel_canvas.pixels[pixel_index] = draw_color;
        }
    }
}
