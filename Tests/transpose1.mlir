"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "transpose_op_test_si32"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]> : tensor<2x3x2xi32>}> : () -> tensor<2x3x2xi32>
    %1 = "stablehlo.transpose"(%0) <{permutation = array<i64: 2, 1, 0>}> : (tensor<2x3x2xi32>) -> tensor<2x3x2xi32>
    "check.expect_eq_const"(%1) <{value = dense<[[[1, 7], [3, 9], [5, 11]], [[2, 8], [4, 10], [6, 12]]]> : tensor<2x3x2xi32>}> : (tensor<2x3x2xi32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

