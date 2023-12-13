const std = @import("std");
const mnist_data_utils = @import("mnist_data_utils.zig");

/// Add ANSI escape codes to around a given string to make it a certain RGB color in the terminal
fn decorateStringWithAnsiColor(
    input_string: []const u8,
    /// Example: `0xFFFFFF`
    optional_foreground_hex_color: ?u24,
    optional_background_hex_color: ?u24,
    allocator: std.mem.Allocator,
) ![]const u8 {
    var foreground_color_code_string: []const u8 = "";
    if (optional_foreground_hex_color) |foreground_hex_color| {
        foreground_color_code_string = try std.fmt.allocPrint(
            allocator,
            "38;2;{d};{d};{d}",
            .{
                // Red channel:
                // Shift the hex color right 16 bits to get the red component all the way down,
                // then make sure we only select the lowest 8 bits by using `& 0xFF`
                (foreground_hex_color >> 16) & 0xFF,
                // Greeen channel:
                // Shift the hex color right 8 bits to get the green component all the way down,
                // then make sure we only select the lowest 8 bits by using `& 0xFF`
                (foreground_hex_color >> 8) & 0xFF,
                // Blue channel:
                // No need to shift the hex color to get the blue component all the way down,
                // but we still need to make sure we only select the lowest 8 bits by using `& 0xFF`
                foreground_hex_color & 0xFF,
            },
        );
    }
    defer allocator.free(foreground_color_code_string);

    var background_color_code_string: []const u8 = "";
    if (optional_background_hex_color) |background_hex_color| {
        background_color_code_string = try std.fmt.allocPrint(
            allocator,
            "48;2;{d};{d};{d}",
            .{
                // Red channel:
                // Shift the hex color right 16 bits to get the red component all the way down,
                // then make sure we only select the lowest 8 bits by using `& 0xFF`
                (background_hex_color >> 16) & 0xFF,
                // Greeen channel:
                // Shift the hex color right 8 bits to get the green component all the way down,
                // then make sure we only select the lowest 8 bits by using `& 0xFF`
                (background_hex_color >> 8) & 0xFF,
                // Blue channel:
                // No need to shift the hex color to get the blue component all the way down,
                // but we still need to make sure we only select the lowest 8 bits by using `& 0xFF`
                background_hex_color & 0xFF,
            },
        );
    }
    defer allocator.free(background_color_code_string);

    var possible_combinator_string: []const u8 = "";
    if (optional_foreground_hex_color != null and optional_background_hex_color != null) {
        possible_combinator_string = ";";
    }

    const string = try std.fmt.allocPrint(
        allocator,
        "\u{001b}[{s}{s}{s}m{s}\u{001b}[0m",
        .{
            foreground_color_code_string,
            possible_combinator_string,
            background_color_code_string,
            input_string,
        },
    );

    return string;
}

pub const TerminalPrintingCharacter = struct {
    character: []const u8,
    /// Some characters render colors differently than others. For example, the full
    /// block character renders colors as-is but the nedium shade block characters render
    /// colors at 1/2 strength, etc respectively.
    opacity_compensation_factor: f64,
};

/// Given a pixel value from 0 to 255, return a unicode block character that represents
/// that pixel value (from nothing, to light shade, to medium shade, to dark shade, to
/// full block).
///
/// We use this in order to facilitate better copy/pasting from the terminal into a
/// plain-text document like a README, vary the characters so they look different from
/// each other.
///
/// See https://en.wikipedia.org/wiki/Block_Elements
fn getCharacterForPixelValue(
    /// Pixel value between 0.0 and 1.0
    pixel_value: f64,
) TerminalPrintingCharacter {
    var character: []const u8 = undefined;
    var opacity_compensation_factor: f64 = 1.0;
    if (pixel_value == 0) {
        // Just a space character that doesn't render anything but still ends up being
        // the same width in a monospace environment.
        character = " ";
        // opacity = 0.0;
        // No need to compensate since this character doesn't render any foreground color
        opacity_compensation_factor = 0.0;
    } else if (pixel_value < 0.25) {
        // Light shade character
        character = "\u{2591}";
        // opacity = 0.25;
        // 1 / 0.25 = 4
        opacity_compensation_factor = 4;
    } else if (pixel_value < 0.5) {
        // Medium shade character
        character = "\u{2592}";
        // opacity = 0.5;
        // 1 / 0.5 = 2
        opacity_compensation_factor = 2;
    } else if (pixel_value < 0.75) {
        // Dark shade character
        character = "\u{2593}";
        // opacity = 0.75;
        // 1 / 0.75 = 1.3333...
        opacity_compensation_factor = @as(f64, 1) / @as(f64, 0.75);
    } else {
        // Full block character
        character = "\u{2588}";
        // opacity = 1;
        // 1 / 1 = 1
        // No need to compensate since anything divided by 1 is itself
        opacity_compensation_factor = 1;
    }

    return .{
        .character = character,
        .opacity_compensation_factor = opacity_compensation_factor,
    };
}

/// Print a representation of a MNIST training/testing image to the terminal using
/// unicode block characters to visualize the pixel values.
pub fn printImage(image: mnist_data_utils.Image, allocator: std.mem.Allocator) !void {
    var width: u8 = 0;
    var height: u8 = 0;
    switch (image) {
        inline else => |case| {
            width = case.width;
            height = case.height;
        },
    }

    std.debug.print("┌", .{});
    for (0..width) |column_index| {
        _ = column_index;
        std.debug.print("──", .{});
    }
    std.debug.print("┐\n", .{});

    for (0..height) |row_index| {
        std.debug.print("│", .{});

        const row_start_index = row_index * width;
        for (0..width) |column_index| {
            const index = row_start_index + column_index;
            var pixel_value: f64 = 0.0;
            switch (image) {
                .raw_image => |raw_image| {
                    pixel_value = @as(f64, @floatFromInt(raw_image.pixels[index])) / 255.0;
                },
                .normalized_image => |normalized_image| {
                    pixel_value = normalized_image.pixels[index];
                },
            }

            const pixel_character = getCharacterForPixelValue(pixel_value);
            const pixel_string = try std.fmt.allocPrint(
                allocator,
                // We use the same character twice to make it look more square (still
                // not perfect though)
                "{0s}{0s}",
                .{pixel_character.character},
            );
            defer allocator.free(pixel_string);

            // Adjust the pixel value to compensate for the opacity of the block
            // character that we're using to represent it.
            const color_channel_value: u8 = @intFromFloat(
                255 * (pixel_value * pixel_character.opacity_compensation_factor),
            );

            const colored_pixel_string = try decorateStringWithAnsiColor(
                pixel_string,
                // Create a white/grey color with the pixel value copied to every color channel.
                (@as(u24, color_channel_value) << 16) |
                    (@as(u24, color_channel_value) << 8) |
                    (@as(u24, color_channel_value) << 0),
                0x000000,
                allocator,
            );
            defer allocator.free(colored_pixel_string);
            std.debug.print("{s}", .{
                colored_pixel_string,
            });
        }
        std.debug.print("│\n", .{});
    }

    std.debug.print("└", .{});
    for (0..width) |column_index| {
        _ = column_index;
        std.debug.print("──", .{});
    }
    std.debug.print("┘\n", .{});
}

/// Print a representation of a MNIST training/testing image (with the expected label)
/// to the terminal using unicode block characters to visualize the pixel values.
pub fn printLabeledImage(labeled_image: mnist_data_utils.LabeledImage, allocator: std.mem.Allocator) !void {
    std.debug.print("┌──────────┐\n", .{});
    std.debug.print("│ Label: {d} │\n", .{labeled_image.label});
    try printImage(labeled_image.image, allocator);
}
