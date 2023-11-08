const DataPoint = @import("../../data_point.zig").DataPoint;

// Binary value can only be 0 or 1
pub const xor_labels = [_]u8{
    0,
    1,
};
pub const XorDataPoint = DataPoint(u8, &xor_labels);
// The XOR data points
pub var xor_data_points = [_]XorDataPoint{
    XorDataPoint.init(&[_]f64{ 0, 0 }, 0),
    XorDataPoint.init(&[_]f64{ 0, 1 }, 1),
    XorDataPoint.init(&[_]f64{ 1, 0 }, 1),
    XorDataPoint.init(&[_]f64{ 1, 1 }, 0),
};
