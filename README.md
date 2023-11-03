# Zig neural network library

This is meant to be a clear, annotated, from scratch, neural network library in
Zig.

To add some buzzword details, it's a multi-layer perceptron (MLP) with backpropagation
(reverse-mode automatic differentiation) and stochastic gradient descent (SGD).

This is heavily inspired by my [first neural network
implementation](https://github.com/MadLittleMods/zig-ocr-neural-network) which was
heavily based on [Sebastian Lague's video](https://www.youtube.com/watch?v=hfMk-kjRv4c)
and now this implementation makes things a bit simpler to understand by following the
pattern from [Omar Aflak's (The Independent Code)
video](https://www.youtube.com/watch?v=pauPCy_s0Ok) where layers just have
`forward(...)`/`backward(...)` methods and the activations are just another layer in the
network (reverse-mode automatic differentiation). See the [*developer
notes*](./dev-notes.md) for more details.


## Usage

Tested with Zig 0.11.0

Compatible with the Zig package manager. Just define it as a dependency in your
`build.zig.zon` file.

TODO: Update to use `zig-neural-network` instead of `zshuffle` once we figure it out

`build.zig.zon`
```zig
.{
    .name = "my-foo-project",
    .version = "0.0.0",
    .dependencies = .{
        .zshuffle = .{
            .url = "https://github.com/madlittlemods/zig-neural-network/archive/abcde.tar.gz",
            .hash = "1220416f31bac21c9f69c2493110064324b2ba9e0257ce0db16fb4f94657124d7abc",
        },
    },
}
```

`build.zig`
```zig
const zshuffle_pkg = b.dependency("zshuffle", .{
    .target = target,
    .optimize = optimize,
});
const zshuffle_mod = zshuffle_pkg.module("zshuffle");
exe.addModule("zshuffle", zshuffle_mod);
exe_tests.addModule("zshuffle", zshuffle_mod);
```


## Example

See TODO

```sh
$ zig build run-xor
```


### Logging

In order to adjust the log level that the Zig neural network library prints out, define
a public `std_options.log_scope_levels` declaration in the root file (where ever main
is), like the following. This library writes to the `.zig_neural_network` scope.

`main.zig`
```zig
pub const std_options = struct {
    pub const log_level = .debug;

    pub const log_scope_levels = &[_]std.log.ScopeLevel{
        .{ .scope = .zig_neural_network, .level = .debug },
        .{ .scope = .library_foo, .level = .warn },
        .{ .scope = .library_bar, .level = .info },
    };
};

// ...
```
