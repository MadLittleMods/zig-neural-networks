const std = @import("std");
const DataPoint = @import("../../data_point.zig").DataPoint;
const convertLabelEnumToOneHotEncodedEnumMap = @import("../../data_point.zig").convertLabelEnumToOneHotEncodedEnumMap;

// Binary value can only be 0 or 1
pub const XorLabel = enum {
    zero,
    one,
};
pub const one_hot_xor_label_map = convertLabelEnumToOneHotEncodedEnumMap(XorLabel);

// FIXME: This function can be deleted after https://github.com/ziglang/zig/pull/18112
// merges and we support a Zig version to use the method directly.
pub fn getPtrConstAssertContains(key: XorLabel) *const [@typeInfo(XorLabel).Enum.fields.len]f64 {
    const opt_one_hot = one_hot_xor_label_map.getPtrConst(key);
    if (opt_one_hot) |one_hot| {
        return one_hot;
    } else {
        std.debug.assert(false, "one_hot_xor_label_map should contain key");
    }
}

// The XOR data points
pub const xor_data_points = [_]DataPoint{
    DataPoint.init(&[_]f64{ 0, 0 }, getPtrConstAssertContains(.zero)),
    DataPoint.init(&[_]f64{ 0, 1 }, getPtrConstAssertContains(.one)),
    DataPoint.init(&[_]f64{ 1, 0 }, getPtrConstAssertContains(.one)),
    DataPoint.init(&[_]f64{ 1, 1 }, getPtrConstAssertContains(.zero)),
};
