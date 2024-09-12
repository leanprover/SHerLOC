"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "slice_op"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[0, 0, 1, 0, 0, 1], [0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 1]]> : tensor<3x6xi64>}> : () -> tensor<3x6xi64>
    %1 = "stablehlo.slice"(%0) <{limit_indices = array<i64: 3, 6>, start_indices = array<i64: 0, 2>, strides = array<i64: 2, 3>}> : (tensor<3x6xi64>) -> tensor<2x2xi64>
    "check.expect_eq_const"(%1) <{value = dense<1> : tensor<2x2xi64>}> : (tensor<2x2xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

