"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "dot_general_op_test_si64"}> ({
    %9 = "stablehlo.constant"() <{value = dense<[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]> : tensor<2x2x2xi64>}> : () -> tensor<2x2x2xi64>
    %10 = "stablehlo.constant"() <{value = dense<[[[1, 0], [0, 1]], [[1, 0], [0, 1]]]> : tensor<2x2x2xi64>}> : () -> tensor<2x2x2xi64>
    %11 = "stablehlo.dot_general"(%9, %10) <{dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]}> : (tensor<2x2x2xi64>, tensor<2x2x2xi64>) -> tensor<2x2x2xi64>
    "check.expect_eq_const"(%11) <{value = dense<[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]> : tensor<2x2x2xi64>}> : (tensor<2x2x2xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "dot_general_op_test_algorithm"}> ({
    %6 = "stablehlo.constant"() <{value = dense<[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]> : tensor<2x2x2xi64>}> : () -> tensor<2x2x2xi64>
    %7 = "stablehlo.constant"() <{value = dense<[[[1, 0], [0, 1]], [[1, 0], [0, 1]]]> : tensor<2x2x2xi64>}> : () -> tensor<2x2x2xi64>
    %8 = "stablehlo.dot_general"(%6, %7) <{algorithm = #stablehlo.dot_algorithm<lhs_precision_type = tf32, rhs_precision_type = tf32, accumulation_type = tf32, lhs_component_count = 1, rhs_component_count = 1, num_primitive_operations = 1, allow_imprecise_accumulation = false>, dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]}> : (tensor<2x2x2xi64>, tensor<2x2x2xi64>) -> tensor<2x2x2xi64>
    "check.expect_eq_const"(%8) <{value = dense<[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]> : tensor<2x2x2xi64>}> : (tensor<2x2x2xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "dot_general_op_test_empty_dims"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[1, 2], [3, 4]]> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
    %4 = "stablehlo.constant"() <{value = dense<[[1, 0], [0, 1]]> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
    %5 = "stablehlo.dot_general"(%3, %4) <{dot_dimension_numbers = #stablehlo.dot<>, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]}> : (tensor<2x2xi64>, tensor<2x2xi64>) -> tensor<2x2x2x2xi64>
    "check.expect_eq_const"(%5) <{value = dense<[[[[1, 0], [0, 1]], [[2, 0], [0, 2]]], [[[3, 0], [0, 3]], [[4, 0], [0, 4]]]]> : tensor<2x2x2x2xi64>}> : (tensor<2x2x2x2xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "dot_general_op_test_different_operand_and_result_element_types"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]], [[5.000000e+00, 6.000000e+00], [7.000000e+00, 8.000000e+00]]]> : tensor<2x2x2xf32>}> : () -> tensor<2x2x2xf32>
    %1 = "stablehlo.constant"() <{value = dense<[[[1.000000e+00, 0.000000e+00], [0.000000e+00, 1.000000e+00]], [[1.000000e+00, 0.000000e+00], [0.000000e+00, 1.000000e+00]]]> : tensor<2x2x2xf32>}> : () -> tensor<2x2x2xf32>
    %2 = "stablehlo.dot_general"(%0, %1) <{dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>}> : (tensor<2x2x2xf32>, tensor<2x2x2xf32>) -> tensor<2x2x2xf64>
    "check.expect_eq_const"(%2) <{value = dense<[[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]], [[5.000000e+00, 6.000000e+00], [7.000000e+00, 8.000000e+00]]]> : tensor<2x2x2xf64>}> : (tensor<2x2x2xf64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

