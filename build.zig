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

    // Build options
    const tracy = b.option([]const u8, "tracy", "Enable Tracy integration (for tracing and profiling). Supply path to locally cloned Tracy source");
    const tracy_callstack = b.option(bool, "tracy_callstack", "Include callstack information with Tracy data. Does nothing if -Dtracy is not provided") orelse (tracy != null);
    const tracy_allocation = b.option(bool, "tracy_allocation", "Include allocation information with Tracy data. Does nothing if -Dtracy is not provided") orelse (tracy != null);

    const build_options = b.addOptions();
    build_options.addOption(bool, "enable_tracy", tracy != null);
    build_options.addOption(bool, "enable_tracy_callstack", tracy_callstack);
    build_options.addOption(bool, "enable_tracy_allocation", tracy_allocation);

    const zshuffle_dep = b.dependency("zshuffle", .{
        .target = target,
        .optimize = optimize,
    });
    const zshuffle_mod = zshuffle_dep.module("zshuffle");

    // We define this mainly so that our sub-dependencies are pulled
    // in when someone uses our library as a dependency.
    //
    // But we also use this internally to build and test our examples
    const module = b.addModule("zig-neural-networks", .{
        .source_file = .{ .path = "src/main.zig" },
        .dependencies = &.{
            // Make the build options available to be imported via
            // `@import("build_options");`
            .{
                .name = "build_options",
                .module = build_options.createModule(),
            },
            // Define what dependencies we rely on. This way when people use the Zig
            // package manager to install our package, they will automatically get the
            // dependencies.
            .{ .name = "zshuffle", .module = zshuffle_mod },
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

        // These are the same `build_options` as above, but we create a new one
        // to avoid `error: file exists in multiple modules` (see below).
        const build_options_for_example = b.addOptions();
        build_options_for_example.addOption(bool, "enable_tracy", tracy != null);
        build_options_for_example.addOption(bool, "enable_tracy_callstack", tracy_callstack);
        build_options_for_example.addOption(bool, "enable_tracy_allocation", tracy_allocation);
        // We only add this to avoid `error: file exists in multiple modules`. This just
        // makes it so that the hash of the generated `options.zig` file is different from
        // the `build_options` below.
        build_options_for_example.addOption(bool, "foobarbaz", false);

        // Make the build options available to be imported
        // `@import("build_options");`
        example_exe.addOptions("build_options", build_options_for_example);
        // Possible alternative:
        // example_exe.addModule("build_options", build_options.createModule());

        // Include whatever is necessary to make tracy work
        if (tracy) |tracy_path| {
            includeTracy(b, example_exe, tracy_path);
        }

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

    // Build/Install our actual library artifact
    // ============================================

    const lib = b.addStaticLibrary(.{
        .name = "zig-neural-networks",
        // In this case the main source file is merely a path, however, in more
        // complicated build scripts, this could be a generated file.
        .root_source_file = .{ .path = "src/main.zig" },
        .target = target,
        .optimize = optimize,
    });
    lib.addOptions("build_options", build_options);
    lib.addModule("zshuffle", zshuffle_mod);

    // Include whatever is necessary to make tracy work
    if (tracy) |tracy_path| {
        includeTracy(b, lib, tracy_path);
    }

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
    unit_tests.addOptions("build_options", build_options);
    unit_tests.addModule("zshuffle", zshuffle_dep.module("zshuffle"));

    // Include whatever is necessary to make tracy work
    if (tracy) |tracy_path| {
        includeTracy(b, unit_tests, tracy_path);
    }

    const run_unit_tests_cmd = b.addRunArtifact(unit_tests);

    // This creates a build step. It will be visible in the `zig build --help` menu,
    // and can be selected like this: `zig build test`
    // This will evaluate the `test` step rather than the default, which is "install".
    const test_step = b.step("test", "Run library tests");
    test_step.dependOn(&run_unit_tests_cmd.step);
}

// (Logic is based on https://github.com/ziglang/zig/blob/0.11.0/build.zig#L349-L371)
fn includeTracy(b: *std.Build, exe: *std.Build.LibExeObjStep, tracy_path: []const u8) void {
    const client_cpp = b.pathJoin(
        &[_][]const u8{ tracy_path, "public", "TracyClient.cpp" },
    );

    // On mingw, we need to opt into windows 7+ to get some features required by tracy.
    const tracy_c_flags: []const []const u8 = if (exe.target.isWindows() and exe.target.getAbi() == .gnu)
        &[_][]const u8{ "-DTRACY_ENABLE=1", "-fno-sanitize=undefined", "-D_WIN32_WINNT=0x601" }
    else
        &[_][]const u8{ "-DTRACY_ENABLE=1", "-fno-sanitize=undefined" };

    exe.addIncludePath(.{ .cwd_relative = tracy_path });
    exe.addCSourceFile(.{ .file = .{ .cwd_relative = client_cpp }, .flags = tracy_c_flags });
    exe.linkLibCpp();
    exe.linkLibC();

    if (exe.target.isWindows()) {
        exe.linkSystemLibrary("dbghelp");
        exe.linkSystemLibrary("ws2_32");
    }
}
