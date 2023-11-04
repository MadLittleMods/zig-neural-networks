const std = @import("std");

/// .ppm
// TODO: There is probably a better way to do this like printing directly a file.
pub fn createPortablePixMap(pixels: []const u24, width: u32, height: u32, allocator: std.mem.Allocator) ![]const u8 {
    var pixel_rows = try std.ArrayList([]const u8).initCapacity(allocator, height);
    defer pixel_rows.deinit();
    for (0..height) |height_index| {
        var pixel_strings = try std.ArrayList([]const u8).initCapacity(allocator, width);
        defer pixel_strings.deinit();
        for (0..width) |width_index| {
            const pixel = pixels[height_index * width + width_index];
            const pixel_string = try std.fmt.allocPrint(
                allocator,
                "{d: >3} {d: >3} {d: >3}",
                .{
                    // Red channel:
                    // Shift the hex color right 16 bits to get the red component all the way down,
                    // then make sure we only select the lowest 8 bits by using `& 0xFF`
                    (pixel >> 16) & 0xFF,
                    // Greeen channel:
                    // Shift the hex color right 8 bits to get the green component all the way down,
                    // then make sure we only select the lowest 8 bits by using `& 0xFF`
                    (pixel >> 8) & 0xFF,
                    // Blue channel:
                    // No need to shift the hex color to get the blue component all the way down,
                    // but we still need to make sure we only select the lowest 8 bits by using `& 0xFF`
                    pixel & 0xFF,
                },
            );

            try pixel_strings.append(pixel_string);
        }

        var pixel_row = try std.mem.join(allocator, "    ", pixel_strings.items);
        try pixel_rows.append(pixel_row);

        for (pixel_strings.items) |pixel_string| {
            allocator.free(pixel_string);
        }
    }

    var pixel_data_string = try std.mem.join(allocator, "\n", pixel_rows.items);
    defer allocator.free(pixel_data_string);

    for (pixel_rows.items) |pixel_row| {
        allocator.free(pixel_row);
    }

    // The magic number identifies the type of file
    const magic_number = "P3";
    const maximum_color_value = 0xFF;
    const ppm_file_contents = std.fmt.allocPrint(
        allocator,
        "{s}\n{d} {d}\n{d}\n{s}",
        .{
            magic_number,
            width,
            height,
            maximum_color_value,
            pixel_data_string,
        },
    );

    return ppm_file_contents;
}
