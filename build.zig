const std = @import("std");

// Although this function looks imperative, note that its job is to
// declaratively construct a build graph that will be executed by an external
// runner.
pub fn build(b: *std.Build) !void {
    // Standard target options allows the person running `zig build` to choose
    // what target to build for. Here we do not override the defaults, which
    // means any target is allowed, and the default is native. Other options
    // for restricting supported target set are available.
    const target = b.standardTargetOptions(.{});

    // Standard optimization options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall. Here we do not
    // set a preferred release mode, allowing the user to decide how to optimize.
    const optimize = b.standardOptimizeOption(.{});

    const zshuffle_dep = b.dependency("zshuffle", .{
        .target = target,
        .optimize = optimize,
    });
    const zshuffle_mod = zshuffle_dep.module("zshuffle");

    const tracy_dep = b.dependency("tracy", .{
        .target = target,
        .optimize = optimize,
    });
    const tracy_mod = tracy_dep.module("tracy");

    // We define this mainly so that our sub-dependencies are pulled
    // in when someone uses our library as a dependency.
    //
    // But we also use this internally to build and test our examples
    const module = b.addModule("zig-neural-networks", .{
        .source_file = .{ .path = "src/main.zig" },
        .dependencies = &.{
            // Define what dependencies we rely on. This way when people use the Zig
            // package manager to install our package, they will automatically get the
            // dependencies.
            .{ .name = "zshuffle", .module = zshuffle_mod },
            .{ .name = "tracy", .module = tracy_mod },
        },
    });

    // Building examples
    // ============================================
    //
    // Based on the zap build: https://github.com/zigzap/zap/blob/8a2d077bd8627c429de4fef3b1899296e6201c0a/build.zig
    const all_step = b.step("all", "build all examples");
    inline for ([_]struct {
        name: []const u8,
        src: []const u8,
    }{
        .{ .name = "xor", .src = "examples/xor/main.zig" },
        .{ .name = "xy_animal_example", .src = "examples/xy_animal_example/main.zig" },
    }) |exe_cfg| {
        const exe_name = exe_cfg.name;
        const exe_src = exe_cfg.src;
        const exe_build_desc = try std.fmt.allocPrint(
            b.allocator,
            "Build the {s} example",
            .{exe_name},
        );
        const exe_run_stepname = try std.fmt.allocPrint(
            b.allocator,
            "run-{s}",
            .{exe_name},
        );
        const exe_run_stepdesc = try std.fmt.allocPrint(
            b.allocator,
            "Run the {s} example",
            .{exe_name},
        );
        const example_step = b.step(exe_name, exe_build_desc);

        const example_exe = b.addExecutable(.{
            .name = exe_name,
            .root_source_file = .{ .path = exe_src },
            .target = target,
            .optimize = optimize,
        });
        // Make the `zig-neural-networks` module available to be imported via `@import("zig-neural-networks")`
        example_exe.addModule("zig-neural-networks", module);

        // install the artifact - depending on the "example"
        const example_build_step = b.addInstallArtifact(example_exe, .{});

        // This *creates* a Run step in the build graph, to be executed when another
        // step is evaluated that depends on it. The next line below will establish
        // such a dependency.
        const example_run_cmd = b.addRunArtifact(example_exe);
        // By making the run step depend on the install step, it will be run from the
        // installation directory rather than directly from within the cache directory.
        // This is not necessary, however, if the application depends on other installed
        // files, this ensures they will be present and in the expected location.
        example_run_cmd.step.dependOn(&example_build_step.step);

        // This allows the user to pass arguments to the application in the build
        // command itself, like this: `zig build run -- arg1 arg2 etc`
        if (b.args) |args| {
            example_run_cmd.addArgs(args);
        }

        // This creates a build step. It will be visible in the `zig build --help` menu,
        // and can be selected like this: `zig build run`
        // This will evaluate the `run` step rather than the default, which is "install".
        const example_run_step = b.step(exe_run_stepname, exe_run_stepdesc);
        example_run_step.dependOn(&example_run_cmd.step);

        example_step.dependOn(&example_build_step.step);
        all_step.dependOn(&example_build_step.step);
    }

    // TODO: Section title
    // ============================================

    const lib = b.addStaticLibrary(.{
        .name = "zig-neural-networks",
        // In this case the main source file is merely a path, however, in more
        // complicated build scripts, this could be a generated file.
        .root_source_file = .{ .path = "src/main.zig" },
        .target = target,
        .optimize = optimize,
    });

    // This declares intent for the library to be installed into the standard
    // location when the user invokes the "install" step (the default step when
    // running `zig build`).
    b.installArtifact(lib);

    // Testing
    // ============================================

    // Creates a step for unit testing. This only builds the test executable
    // but does not run it.
    const unit_tests = b.addTest(.{
        .root_source_file = .{ .path = "src/tests.zig" },
        .target = target,
        .optimize = optimize,
    });
    unit_tests.addModule("zshuffle", zshuffle_dep.module("zshuffle"));
    unit_tests.addModule("tracy", tracy_dep.module("tracy"));

    const run_unit_tests_cmd = b.addRunArtifact(unit_tests);

    // This creates a build step. It will be visible in the `zig build --help` menu,
    // and can be selected like this: `zig build test`
    // This will evaluate the `test` step rather than the default, which is "install".
    const test_step = b.step("test", "Run library tests");
    test_step.dependOn(&run_unit_tests_cmd.step);
}
