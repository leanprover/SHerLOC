"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "dot_general_op_test_si64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]> : tensor<2x2x2xi64>}> : () -> tensor<2x2x2xi64>
    %1 = "stablehlo.constant"() <{value = dense<[[[1, 0], [0, 1]], [[1, 0], [0, 1]]]> : tensor<2x2x2xi64>}> : () -> tensor<2x2x2xi64>
    %2 = "stablehlo.dot_general"(%0, %1) <{dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]}> : (tensor<2x2x2xi64>, tensor<2x2x2xi64>) -> tensor<2x2x2xi64>
    "check.expect_eq_const"(%2) <{value = dense<[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]> : tensor<2x2x2xi64>}> : (tensor<2x2x2xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "dot_general_op_test_algorithm"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]> : tensor<2x2x2xi64>}> : () -> tensor<2x2x2xi64>
    %1 = "stablehlo.constant"() <{value = dense<[[[1, 0], [0, 1]], [[1, 0], [0, 1]]]> : tensor<2x2x2xi64>}> : () -> tensor<2x2x2xi64>
    %2 = "stablehlo.dot_general"(%0, %1) <{algorithm = #stablehlo.dot_algorithm<lhs_precision_type = tf32, rhs_precision_type = tf32, accumulation_type = f32, lhs_component_count = 1, rhs_component_count = 1, num_primitive_operations = 1, allow_imprecise_accumulation = false>, dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]}> : (tensor<2x2x2xi64>, tensor<2x2x2xi64>) -> tensor<2x2x2xi64>
    "check.expect_eq_const"(%2) <{value = dense<[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]> : tensor<2x2x2xi64>}> : (tensor<2x2x2xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "dot_general_op_test_empty_dims"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[1, 2], [3, 4]]> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
    %1 = "stablehlo.constant"() <{value = dense<[[1, 0], [0, 1]]> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
    %2 = "stablehlo.dot_general"(%0, %1) <{dot_dimension_numbers = #stablehlo.dot<>, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]}> : (tensor<2x2xi64>, tensor<2x2xi64>) -> tensor<2x2x2x2xi64>
    "check.expect_eq_const"(%2) <{value = dense<[[[[1, 0], [0, 1]], [[2, 0], [0, 2]]], [[[3, 0], [0, 3]], [[4, 0], [0, 4]]]]> : tensor<2x2x2x2xi64>}> : (tensor<2x2x2x2xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "dot_general_op_test_different_operand_and_result_element_types"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]], [[5.000000e+00, 6.000000e+00], [7.000000e+00, 8.000000e+00]]]> : tensor<2x2x2xf32>}> : () -> tensor<2x2x2xf32>
    %1 = "stablehlo.constant"() <{value = dense<[[[1.000000e+00, 0.000000e+00], [0.000000e+00, 1.000000e+00]], [[1.000000e+00, 0.000000e+00], [0.000000e+00, 1.000000e+00]]]> : tensor<2x2x2xf32>}> : () -> tensor<2x2x2xf32>
    %2 = "stablehlo.dot_general"(%0, %1) <{dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>}> : (tensor<2x2x2xf32>, tensor<2x2x2xf32>) -> tensor<2x2x2xf64>
    "check.expect_eq_const"(%2) <{value = dense<[[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]], [[5.000000e+00, 6.000000e+00], [7.000000e+00, 8.000000e+00]]]> : tensor<2x2x2xf64>}> : (tensor<2x2x2xf64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "add_op_test_f8E3M4"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<4xf8E3M4>}> : () -> tensor<4xf8E3M4>
    %1 = "stablehlo.dot_general"(%0, %0) <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>}> : (tensor<4xf8E3M4>, tensor<4xf8E3M4>) -> tensor<f8E3M4>
    "check.expect_almost_eq_const"(%1) <{value = dense<1.400000e+01> : tensor<f8E3M4>}> : (tensor<f8E3M4>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "add_op_test_f8E4M3"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00]> : tensor<8xf8E4M3>}> : () -> tensor<8xf8E4M3>
    %1 = "stablehlo.dot_general"(%0, %0) <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>}> : (tensor<8xf8E4M3>, tensor<8xf8E4M3>) -> tensor<f8E4M3>
    "check.expect_almost_eq_const"(%1) <{value = dense<1.440000e+02> : tensor<f8E4M3>}> : (tensor<f8E4M3>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

