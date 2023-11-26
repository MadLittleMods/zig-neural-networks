const std = @import("std");
const DataPoint = @import("../../data_point.zig").DataPoint;
const convertLabelEnumToOneHotEncodedEnumMap = @import("../../data_point.zig").convertLabelEnumToOneHotEncodedEnumMap;

// Binary value can only be 0 or 1
pub const XorLabel = enum {
    zero,
    one,
};
pub const one_hot_xor_label_map = convertLabelEnumToOneHotEncodedEnumMap(XorLabel);

// The XOR data points
pub const xor_data_points = [_]DataPoint{
    // FIXME: Once https://github.com/ziglang/zig/pull/18112 merges and we support a Zig
    // version that includes it, we should use `getPtrConstAssertContains(...)` instead.
    DataPoint.init(&[_]f64{ 0, 0 }, one_hot_xor_label_map.getPtrConst(.zero).?),
    DataPoint.init(&[_]f64{ 0, 1 }, one_hot_xor_label_map.getPtrConst(.one).?),
    DataPoint.init(&[_]f64{ 1, 0 }, one_hot_xor_label_map.getPtrConst(.one).?),
    DataPoint.init(&[_]f64{ 1, 1 }, one_hot_xor_label_map.getPtrConst(.zero).?),
};
