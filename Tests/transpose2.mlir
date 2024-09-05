"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "transpose_op_test_si32"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]> : tensor<2x3x2xi32>}> : () -> tensor<2x3x2xi32>
    %1 = "stablehlo.transpose"(%0) <{permutation = array<i64: 2, 1, 0>}> : (tensor<2x3x2xi32>) -> tensor<2x3x2xi32>
    %2 = "stablehlo.transpose"(%1) <{permutation = array<i64: 2, 1, 0>}> : (tensor<2x3x2xi32>) -> tensor<2x3x2xi32>
    "check.expect_eq_const"(%2) <{value = dense<[[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]> : tensor<2x3x2xi32>}> : (tensor<2x3x2xi32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

