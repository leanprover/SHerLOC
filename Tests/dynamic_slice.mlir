"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "dynamic_slice"}> ({
    %0 = "stablehlo.constant"() <{value = dense<1> : tensor<3x3xi64>}> : () -> tensor<3x3xi64>
    %1 = "stablehlo.constant"() <{value = dense<3> : tensor<i64>}> : () -> tensor<i64>
    %2 = "stablehlo.constant"() <{value = dense<3> : tensor<i64>}> : () -> tensor<i64>
    %3 = "stablehlo.dynamic_slice"(%0, %1, %2) <{slice_sizes = array<i64: 3, 3>}> : (tensor<3x3xi64>, tensor<i64>, tensor<i64>) -> tensor<3x3xi64>
    "check.expect_eq_const"(%3) <{value = dense<1> : tensor<3x3xi64>}> : (tensor<3x3xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

