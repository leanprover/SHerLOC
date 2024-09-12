"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "convolution_op_test_si64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[[1], [2], [5], [6]], [[3], [4], [7], [8]], [[10], [11], [14], [15]], [[12], [13], [16], [17]]]]> : tensor<1x4x4x1xi64>}> : () -> tensor<1x4x4x1xi64>
    %1 = "stablehlo.constant"() <{value = dense<1> : tensor<3x3x1x1xi64>}> : () -> tensor<3x3x1x1xi64>
    %2 = "stablehlo.convolution"(%0, %1) <{batch_group_count = 1 : i64, dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>, feature_group_count = 1 : i64, lhs_dilation = array<i64: 2, 2>, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>], window_strides = array<i64: 4, 4>}> : (tensor<1x4x4x1xi64>, tensor<3x3x1x1xi64>) -> tensor<1x2x2x1xi64>
    "check.expect_eq_const"(%2) <{value = dense<[[[[10], [26]], [[46], [62]]]]> : tensor<1x2x2x1xi64>}> : (tensor<1x2x2x1xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "convolution_op_test_padding"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[[1], [2], [5], [6]], [[3], [4], [7], [8]], [[10], [11], [14], [15]], [[12], [13], [16], [17]]]]> : tensor<1x4x4x1xi64>}> : () -> tensor<1x4x4x1xi64>
    %1 = "stablehlo.constant"() <{value = dense<1> : tensor<3x3x1x1xi64>}> : () -> tensor<3x3x1x1xi64>
    %2 = "stablehlo.convolution"(%0, %1) <{batch_group_count = 1 : i64, dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>, feature_group_count = 1 : i64, lhs_dilation = array<i64: 2, 2>, padding = dense<1> : tensor<2x2xi64>, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>], window_strides = array<i64: 4, 4>}> : (tensor<1x4x4x1xi64>, tensor<3x3x1x1xi64>) -> tensor<1x2x2x1xi64>
    "check.expect_eq_const"(%2) <{value = dense<[[[[1], [5]], [[10], [14]]]]> : tensor<1x2x2x1xi64>}> : (tensor<1x2x2x1xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "convolution_batch_group_count_4"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[[1], [2], [5], [6]], [[3], [4], [7], [8]], [[10], [11], [14], [15]], [[12], [13], [16], [17]]]]> : tensor<1x4x4x1xi64>}> : () -> tensor<1x4x4x1xi64>
    %1 = "stablehlo.constant"() <{value = dense<1> : tensor<1x2x1x4xi64>}> : () -> tensor<1x2x1x4xi64>
    %2 = "stablehlo.convolution"(%0, %1) <{batch_group_count = 4 : i64, dimension_numbers = #stablehlo.conv<[0, b, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>, feature_group_count = 1 : i64, lhs_dilation = array<i64: 2, 2>, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>], window_strides = array<i64: 4, 4>}> : (tensor<1x4x4x1xi64>, tensor<1x2x1x4xi64>) -> tensor<1x1x2x4xi64>
    "check.expect_eq_const"(%2) <{value = dense<[[[[1, 3, 10, 12], [5, 7, 14, 16]]]]> : tensor<1x1x2x4xi64>}> : (tensor<1x1x2x4xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "convolution_feature_group_count_2"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[[1], [2], [5], [6]], [[3], [4], [7], [8]], [[10], [11], [14], [15]], [[12], [13], [16], [17]]]]> : tensor<1x4x4x1xi64>}> : () -> tensor<1x4x4x1xi64>
    %1 = "stablehlo.constant"() <{value = dense<1> : tensor<1x2x1x4xi64>}> : () -> tensor<1x2x1x4xi64>
    %2 = "stablehlo.convolution"(%0, %1) <{batch_group_count = 1 : i64, dimension_numbers = #stablehlo.conv<[b, 0, f, 1]x[0, i, 1, o]->[b, 0, 1, f]>, feature_group_count = 2 : i64, lhs_dilation = array<i64: 2, 2>, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>], window_strides = array<i64: 4, 4>}> : (tensor<1x4x4x1xi64>, tensor<1x2x1x4xi64>) -> tensor<1x2x1x4xi64>
    "check.expect_eq_const"(%2) <{value = dense<[[[[3, 3, 11, 11]], [[21, 21, 29, 29]]]]> : tensor<1x2x1x4xi64>}> : (tensor<1x2x1x4xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

