# Zig neural networks library

This is meant to be a understandable, annotated, from scratch, neural network library in
Zig.

To add some buzzword details, it's a multi-layer perceptron (MLP) with backpropagation
and stochastic gradient descent (SGD). Optional momentum, ...

Performance-wise, it should be just fine for your small application-specific purposes.
This library currently avoids the pesky vector/matrix libraries which can make it hard
to follow what exactly is being multiplied together (just uses flat arrays) when you're
trying to wrap your head around the concepts. If we ever decide to use one of
vector/matrix library, I plan to keep around the "slow" variants of the forward/backward
methods alongside the optimized versions.

This is heavily inspired by my [first neural network
implementation](https://github.com/MadLittleMods/zig-ocr-neural-network) which was based
on [Sebastian Lague's video](https://www.youtube.com/watch?v=hfMk-kjRv4c) and now this
library implementation makes things a bit simpler to understand (at least math-wise) by
following a pattern from [Omar Aflak's (The Independent Code)
video](https://www.youtube.com/watch?v=pauPCy_s0Ok) where activation functions are just
another layer in the network. See the [*developer notes*](./dev-notes.md) for more
details.

Or if you're curious about how the math equations/formulas are derived, check out the
[*developer notes*](./dev-notes.md#math) for more details.


## Installation

Tested with Zig 0.11.0

Compatible with the Zig package manager. Just define it as a dependency in your
`build.zig.zon` file.

`build.zig.zon`
```zig
.{
    .name = "my-foo-project",
    .version = "0.0.0",
    .dependencies = .{
        .@"zig-neural-networks" = .{
            .url = "https://github.com/madlittlemods/zig-neural-networks/archive/<some-commit-hash-abcde>.tar.gz",
            .hash = "1220416f31bac21c9f69c2493110064324b2ba9e0257ce0db16fb4f94657124d7abc",
        },
    },
}
```

`build.zig`
```zig
const neural_networks_pkg = b.dependency("zig-neural-networks", .{
    .target = target,
    .optimize = optimize,
});
const neural_networks_mod = neural_networks_pkg.module("zig-neural-networks");
// Make the `zig-neural-networks` module available to be imported via `@import("zig-neural-networks")`
exe.addModule("zig-neural-networks", neural_networks_mod);
exe_tests.addModule("zig-neural-networks", neural_networks_mod);
```


## Examples

#### MNIST OCR digit recognition

Setup: Download and extract the MNIST dataset from http://yann.lecun.com/exdb/mnist/ to
a data directory in the mnist example project, `examples/mnist/data/`. Here is a
copy-paste command you can run:

```sh
# Make a data/ directory
mkdir examples/mnist/data/ &&
# Move to the data/ directory
cd examples/mnist/data/ &&
# Download the MNIST dataset
curl -O http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz &&
curl -O http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz &&
curl -O http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz &&
curl -O http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz &&
# Unzip the files
gunzip *.gz &&
# Move back to the root of the project
cd ../../../
```

Then run the example:

With the MNIST OCR example, on my machine, I can complete 1 epoch of training in ~1
minute which gets to 94% accuracy and creeps to 97% after a few more epochs (60k
training images, 10k test images):

```sh
$ zig build run-mnist
debug: Created normalized data points. Training on 60000 data points, testing on 10000
debug: Here is what the first training data point looks like:
┌──────────┐
│ Label: 5 │
┌────────────────────────────────────────────────────────┐
│                                                        │
│                                                        │
│                                                        │
│                                                        │
│                                                        │
│                        ░░░░░░░░▒▒▓▓▓▓░░▓▓████▒▒        │
│                ░░░░▒▒▓▓▓▓████████████▓▓██████▒▒        │
│              ░░████████████████████▒▒▒▒▒▒░░░░          │
│              ░░██████████████▓▓████                    │
│                ▒▒▓▓▒▒██████░░  ░░▓▓                    │
│                  ░░░░▓▓██▒▒                            │
│                      ▓▓██▓▓░░                          │
│                      ░░▓▓██▒▒                          │
│                        ░░████▓▓▒▒░░                    │
│                          ▒▒██████▒▒░░                  │
│                            ░░▓▓████▓▓░░                │
│                              ░░▒▒████▓▓                │
│                                  ██████▒▒              │
│                            ░░▓▓▓▓██████░░              │
│                        ░░▓▓██████████▓▓                │
│                    ░░▒▒████████████▒▒                  │
│                ░░▒▒████████████▒▒░░                    │
│            ░░▓▓████████████▒▒░░                        │
│        ░░▓▓████████████▓▓░░                            │
│        ▓▓████████▓▓▓▓░░                                │
│                                                        │
│                                                        │
│                                                        │
└────────────────────────────────────────────────────────┘
debug: epoch 0   batch 0             3s -> cost 331.64265899045563, accuracy with 100 test points 0.11
debug: epoch 0   batch 5             4s -> cost 242.16033395427667, accuracy with 100 test points 0.56
debug: epoch 0   batch 10            5s -> cost 155.62913461977217, accuracy with 100 test points 0.7
debug: epoch 0   batch 15            5s -> cost 118.45908401769115, accuracy with 100 test points 0.75
[...]
```


#### Simple animal example

This is a small dataset that I made up to test the neural network. There are only 2
arbitrary features (x and y) where the labeled data points (fish and goat) occupy
distinct parts of the graph. Since there are only 2 input features (which means 2
dimensions), we can easily graph the neural network's decision/classification boundary.
It's a good way to visualize the neural network and see how it evolves while training.

This example produces an image called `simple_xy_animal_graph.ppm` every 1,000 epochs
showing the decision/classification boundary.

```sh
$ zig build run-xy_animal_example
```

![](https://github.com/MadLittleMods/zig-neural-networks/assets/558581/caaf35ed-cb45-4f9c-bdbe-12418ecff5a7)


#### Barebones XOR example

There is also a barebones XOR example (only 4 possible data points) which just trains a neural
network to act like a XOR ("exclusive or") gate.

```sh
$ zig build run-xor
```

<details>
<summary>Boundary graph</summary>

![](https://github.com/MadLittleMods/zig-neural-networks/assets/558581/d22817a6-1439-4b6c-9e43-bafd28cf5d19)

</details>


#### Using custom layer types

Perhaps you want to implement and use a custom dropout layer, skip layer, or
[convolutional/reshape](https://www.youtube.com/watch?v=Lakz2MoHy6o) layer. Since the
`Layer` type is just an interface, you can implement your own layer types in Zig and
pass them to the neural network.

[`examples/mnist/main_custom.zig`](https://github.com/MadLittleMods/zig-neural-networks/blob/main/examples/mnist/main_custom.zig)
```sh
$ zig build run-mnist_custom
```



### Saving and loading (serialize/deserialize)

You can save and load a `NeuralNetwork` with the standard Zig `std.json` methods.
This is useful so you can save all of your training progress and load it back up later
to continue training or to use the network to predict/classify data in your real
application.

```zig
var neural_network = neural_networks.NeuralNetwork.initFromLayers(
    // ...
);

// Serialize the neural network
const serialized_neural_network = try std.json.stringifyAlloc(
    allocator,
    neural_network,
    .{ .whitespace = .indent_2 },
);
defer allocator.free(serialized_neural_network);

// Deserialize the neural network
const parsed_neural_network = try std.json.parseFromSlice(
    NeuralNetwork,
    allocator,
    serialized_neural_network,
    .{},
);
defer parsed_neural_network.deinit();
const deserialized_neural_network = parsed_neural_network.value;
// Use `deserialized_neural_network` as desired
```

You can also see how this works in each of the examples where they save the state of the
`NeuralNetwork` out to a checkpoint file as it trains.

The MNIST example even has some resume training functionality to parse/load/deserialize
the JSON checkpoint file back to a `NeuralNetwork` to use: `zig build run-mnist --
--resume-training-from-last-checkpoint`


### Logging

In order to adjust the log level that the Zig neural network library prints out, define
a public `std_options.log_scope_levels` declaration in the root file (where ever main
is), like the following. This library writes to the `.zig_neural_networks` scope.

`main.zig`
```zig
pub const std_options = struct {
    pub const log_level = .debug;

    pub const log_scope_levels = &[_]std.log.ScopeLevel{
        .{ .scope = .zig_neural_networks, .level = .debug },
        .{ .scope = .library_foo, .level = .warn },
        .{ .scope = .library_bar, .level = .info },
    };
};

// ...
```


### Tests

Alongside normal tests to ensure the neural network can learn, predict, and classify
data points, the codebase also has gradient checks to ensure that the backpropagation
algorithm is working correctly and slope checks to ensure that the activation and cost
functions and derivatives are accurate and correlated.

```sh
$ zig build test
```
