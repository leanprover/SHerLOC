"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "broadcast_in_dim"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[1], [2], [3]]> : tensor<3x1xi64>}> : () -> tensor<3x1xi64>
    %1 = "stablehlo.constant"() <{value = dense<[3, 2, 2]> : tensor<3xi64>}> : () -> tensor<3xi64>
    %2 = "stablehlo.dynamic_broadcast_in_dim"(%0, %1) <{broadcast_dimensions = array<i64: 0, 2>, known_expanding_dimensions = array<i64: 1>}> {known_non_expanding_dimensions = array<i64: 0>} : (tensor<3x1xi64>, tensor<3xi64>) -> tensor<3x2x2xi64>
    "check.expect_eq_const"(%2) <{value = dense<[[[1, 1], [1, 1]], [[2, 2], [2, 2]], [[3, 3], [3, 3]]]> : tensor<3x2x2xi64>}> : (tensor<3x2x2xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

