"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "dynamic_conv_op_test_si64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[[1], [2], [5], [6]], [[3], [4], [7], [8]], [[10], [11], [14], [15]], [[12], [13], [16], [17]]]]> : tensor<1x4x4x1xi64>}> : () -> tensor<1x4x4x1xi64>
    %1 = "stablehlo.constant"() <{value = dense<1> : tensor<3x3x1x1xi64>}> : () -> tensor<3x3x1x1xi64>
    %2 = "stablehlo.constant"() <{value = dense<1> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
    %3 = "stablehlo.dynamic_conv"(%0, %1, %2) <{batch_group_count = 1 : i64, dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>, feature_group_count = 1 : i64, lhs_dilation = array<i64: 2, 2>, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>], rhs_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 4, 4>}> : (tensor<1x4x4x1xi64>, tensor<3x3x1x1xi64>, tensor<2x2xi64>) -> tensor<1x2x2x1xi64>
    "check.expect_eq_const"(%3) <{value = dense<[[[[1], [5]], [[10], [14]]]]> : tensor<1x2x2x1xi64>}> : (tensor<1x2x2x1xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

