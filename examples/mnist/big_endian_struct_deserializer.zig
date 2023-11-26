const std = @import("std");

/// Given a reader to a binary blob of data encoded in big endian, this function will
/// deserialize the data into the struct given.
///
/// If you already have a buffer of data, you can use `TODO` to create a reader in order
/// to pass it in here.
///
/// Based on https://nathancraddock.com/blog/2022/deserialization-with-zig-metaprogramming/#writing-a-struct-deserializer-with-zig
pub fn BigEndianStructDeserializer(comptime ReaderType: type) type {
    return struct {
        reader: ReaderType,

        // We use our own `readStruct(...)` function instead of `self.reader.readStruct(...)`
        // because the standard library uses a byte swap function that does not work on arrays.
        fn readStruct(self: *const @This(), comptime T: type) !T {
            const fields = std.meta.fields(T);

            var item: T = undefined;
            inline for (fields) |field| {
                @field(item, field.name) = try self.read(field.type);
            }

            return item;
        }

        pub fn read(self: *const @This(), comptime T: type) !T {
            return switch (@typeInfo(T)) {
                .Int => try self.reader.readIntBig(T),
                .Float => try self.reader.readVarInt(T, .Big, @sizeOf(T)),
                .Array => |array| {
                    var arr: [array.len]array.child = undefined;
                    var index: usize = 0;
                    while (index < array.len) : (index += 1) {
                        arr[index] = try self.read(array.child);
                    }
                    return arr;
                },
                .Struct => try self.readStruct(T),
                else => @compileError("unsupported type"),
            };
        }
    };
}

pub fn bigEndianStructDeserializer(reader: anytype) BigEndianStructDeserializer(@TypeOf(reader)) {
    return .{ .reader = reader };
}
