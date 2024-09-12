"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "dynamic_update_slice"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[1, 1, 1, 1], [1, 1, 1, 1], [1, 2, 2, 2], [1, 2, 2, 2]]> : tensor<4x4xi64>}> : () -> tensor<4x4xi64>
    %1 = "stablehlo.constant"() <{value = dense<1> : tensor<2x3xi64>}> : () -> tensor<2x3xi64>
    %2 = "stablehlo.constant"() <{value = dense<4> : tensor<i64>}> : () -> tensor<i64>
    %3 = "stablehlo.constant"() <{value = dense<4> : tensor<i64>}> : () -> tensor<i64>
    %4 = "stablehlo.dynamic_update_slice"(%0, %1, %2, %3) : (tensor<4x4xi64>, tensor<2x3xi64>, tensor<i64>, tensor<i64>) -> tensor<4x4xi64>
    "check.expect_eq_const"(%4) <{value = dense<1> : tensor<4x4xi64>}> : (tensor<4x4xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

