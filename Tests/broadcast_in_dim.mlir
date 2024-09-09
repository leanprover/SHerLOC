"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "broadcast_in_dim"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[1], [2], [3]]> : tensor<3x1xi64>}> : () -> tensor<3x1xi64>
    %1 = "stablehlo.broadcast_in_dim"(%0) <{broadcast_dimensions = array<i64: 0, 2>}> : (tensor<3x1xi64>) -> tensor<3x2x2xi64>
    "check.expect_eq_const"(%1) <{value = dense<[[[1, 1], [1, 1]], [[2, 2], [2, 2]], [[3, 3], [3, 3]]]> : tensor<3x2x2xi64>}> : (tensor<3x2x2xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

