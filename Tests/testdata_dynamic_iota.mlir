"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "dynamic_iota_op_test_si64_dim_0"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[3, 4]> : tensor<2xi64>}> : () -> tensor<2xi64>
    %1 = "stablehlo.dynamic_iota"(%0) <{iota_dimension = 0 : i64}> : (tensor<2xi64>) -> tensor<3x4xi64>
    "check.expect_eq_const"(%1) <{value = dense<[[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]]> : tensor<3x4xi64>}> : (tensor<3x4xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "dynamic_iota_op_test_si64_dim_1"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[3, 4]> : tensor<2xi64>}> : () -> tensor<2xi64>
    %1 = "stablehlo.dynamic_iota"(%0) <{iota_dimension = 1 : i64}> : (tensor<2xi64>) -> tensor<3x4xi64>
    "check.expect_eq_const"(%1) <{value = dense<[[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]> : tensor<3x4xi64>}> : (tensor<3x4xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

