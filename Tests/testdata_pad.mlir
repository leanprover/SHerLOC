"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "pad"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 5, 6, 0], [0, 0, 0, 0]]> : tensor<5x4xi64>}> : () -> tensor<5x4xi64>
    %1 = "stablehlo.constant"() <{value = dense<-1> : tensor<i64>}> : () -> tensor<i64>
    %2 = "stablehlo.pad"(%0, %1) <{edge_padding_high = array<i64: 1, -1>, edge_padding_low = array<i64: 1, -1>, interior_padding = array<i64: 0, 1>}> : (tensor<5x4xi64>, tensor<i64>) -> tensor<7x5xi64>
    "check.expect_eq_const"(%2) <{value = dense<[[-1, -1, -1, -1, -1], [-1, 0, -1, 0, -1], [-1, 1, -1, 2, -1], [-1, 3, -1, 4, -1], [-1, 5, -1, 6, -1], [-1, 0, -1, 0, -1], [-1, -1, -1, -1, -1]]> : tensor<7x5xi64>}> : (tensor<7x5xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

