const std = @import("std");
const neural_networks = @import("zig-neural-networks");

const checkpoint_file_name_prefix: []const u8 = "mnist_neural_network_checkpoint_epoch_";
const json_file_suffix = ".json";
const bytes_per_mb: usize = 1024 * 1024;

/// Saves the the current state of the neural network to a JSON checkpoint file in the
/// root of the project.
///
/// To load and deserialize a neural network from a checkpoint file, you can use
/// `std.json.parseFromSlice(...)` or whatever method from the Zig standard library to
/// parse JSON.
pub fn saveNeuralNetworkCheckpoint(
    neural_network: *neural_networks.NeuralNetwork,
    current_epoch_index: usize,
    allocator: std.mem.Allocator,
) !void {
    // Figure out the path to save the file to in the root directory of the project
    const source_path = std.fs.path.dirname(@src().file) orelse ".";
    const project_root_path = try std.fs.path.resolvePosix(allocator, &[_][]const u8{
        source_path, "../",
    });
    defer allocator.free(project_root_path);
    const file_path = try std.fmt.allocPrint(
        allocator,
        "{s}/{s}{d}.json",
        .{
            // Prepend the project directory path
            project_root_path,
            // Assemble the file name
            checkpoint_file_name_prefix,
            current_epoch_index,
        },
    );
    defer allocator.free(file_path);
    std.log.debug("Saving neural network checkpoint to {s}", .{file_path});

    // Turn the neural network into a string of JSON
    const serialized_neural_network = try std.json.stringifyAlloc(
        allocator,
        neural_network,
        .{
            // To make the JSON more readable and pretty-print
            // .whitespace = .indent_2,
        },
    );
    defer allocator.free(serialized_neural_network);

    // Save the JSON file to disk
    const file = try std.fs.cwd().createFile(file_path, .{});
    defer file.close();
    try file.writeAll(serialized_neural_network);
}

const CheckpointFileInfo = struct {
    file_path: []const u8,
    epoch_index: u32,
};

/// Finds the latest checkpoint file in the root of the project.
pub fn findLatestNeuralNetworkCheckpoint(
    allocator: std.mem.Allocator,
) !CheckpointFileInfo {
    const source_path = std.fs.path.dirname(@src().file) orelse ".";
    const project_root_path = try std.fs.path.resolvePosix(allocator, &[_][]const u8{
        source_path, "../",
    });
    defer allocator.free(project_root_path);
    var iter_dir = try std.fs.cwd().openIterableDir(project_root_path, .{});
    defer iter_dir.close();

    // Find the latest checkpoint file (largest epoch index)
    var latest_epoch_index: u32 = 0;
    var opt_latest_file_name: ?[]const u8 = null;
    var iter_dir_iterator = iter_dir.iterate();
    while (try iter_dir_iterator.next()) |entry| {
        if (entry.kind == .file and
            entry.name.len > (checkpoint_file_name_prefix.len + json_file_suffix.len) and
            std.mem.eql(u8, entry.name[0..checkpoint_file_name_prefix.len], checkpoint_file_name_prefix))
        {
            const number_suffix = entry.name[checkpoint_file_name_prefix.len..(entry.name.len - json_file_suffix.len)];
            const parsed_epoch_index = try std.fmt.parseInt(u32, number_suffix, 10);
            if (parsed_epoch_index >= latest_epoch_index) {
                latest_epoch_index = parsed_epoch_index;
                opt_latest_file_name = entry.name;
            }
        }
    }

    const latest_file_name = if (opt_latest_file_name) |latest_file_name| blk: {
        break :blk latest_file_name;
    } else {
        std.log.err("No neural network checkpoints found in {s}", .{project_root_path});
        return error.NoCheckpointsFound;
    };

    const file_path = try std.fmt.allocPrint(
        allocator,
        "{s}/{s}",
        .{
            // Prepend the project directory path
            project_root_path,
            // And the latest file name
            latest_file_name,
        },
    );
    // defer allocator.free(file_path);

    return .{
        .file_path = file_path,
        .epoch_index = latest_epoch_index,
    };
}

/// Loads the checkpoint file from the project root directory.
pub fn loadNeuralNetworkCheckpoint(
    checkpoint_file_path: []const u8,
    allocator: std.mem.Allocator,
) !std.json.Parsed(neural_networks.NeuralNetwork) {
    std.log.debug("Loading neural network checkpoint from {s}", .{checkpoint_file_path});

    // Get the file contents
    const file = try std.fs.cwd().openFile(checkpoint_file_path, .{});
    defer file.close();
    const file_contents = try file.readToEndAlloc(allocator, 4 * bytes_per_mb);
    defer allocator.free(file_contents);

    // Deserialize the JSON into a neural network
    var scanner = std.json.Scanner.initCompleteInput(allocator, file_contents);
    defer scanner.deinit();
    var diagnostics = std.json.Diagnostics{};
    scanner.enableDiagnostics(&diagnostics);
    // We could alternatively simply use `std.json.parseFromSlice(...)` but we don't get
    // very good error messages from it since there are no diagnostics which tell us
    // what line number caused the problem.
    var parsed_neural_network = std.json.parseFromTokenSource(
        neural_networks.NeuralNetwork,
        allocator,
        &scanner,
        .{},
    ) catch |err| {
        std.log.err("NeuralNetwork JSON parse failed {}, line {}:{}", .{
            err,
            diagnostics.getLine(),
            diagnostics.getColumn(),
        });
        return err;
    };
    return parsed_neural_network;
}
