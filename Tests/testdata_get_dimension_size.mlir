"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "get_dimension_size_op_test_si64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi64>}> : () -> tensor<2x3xi64>
    %1 = "stablehlo.get_dimension_size"(%0) <{dimension = 1 : i64}> : (tensor<2x3xi64>) -> tensor<i32>
    "check.expect_eq_const"(%1) <{value = dense<3> : tensor<i32>}> : (tensor<i32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

