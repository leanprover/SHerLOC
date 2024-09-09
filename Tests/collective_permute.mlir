"builtin.module"() ({
  "builtin.module"() <{sym_name = "cross_replica"}> ({
    "func.func"() <{function_type = (tensor<2x2xi64>) -> tensor<2x2xi64>, sym_name = "collective_permute"}> ({
    ^bb0(%arg1: tensor<2x2xi64>):
      %9 = "stablehlo.collective_permute"(%arg1) <{channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>, source_target_pairs = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>}> : (tensor<2x2xi64>) -> tensor<2x2xi64>
      "func.return"(%9) : (tensor<2x2xi64>) -> ()
    }) : () -> ()
    "func.func"() <{function_type = () -> (), sym_name = "main"}> ({
      %5 = "stablehlo.constant"() <{value = dense<[[1, 2], [3, 4]]> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
      %6 = "stablehlo.constant"() <{value = dense<[[5, 6], [7, 8]]> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
      %7 = "stablehlo.constant"() <{value = dense<[[9, 10], [11, 12]]> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
      %8:3 = "interpreter.run_parallel"(%5, %6, %7) <{programs = [[@collective_permute], [@collective_permute], [@collective_permute]]}> : (tensor<2x2xi64>, tensor<2x2xi64>, tensor<2x2xi64>) -> (tensor<2x2xi64>, tensor<2x2xi64>, tensor<2x2xi64>)
      "check.expect_eq_const"(%8#0) <{value = dense<0> : tensor<2x2xi64>}> : (tensor<2x2xi64>) -> ()
      "check.expect_eq_const"(%8#1) <{value = dense<[[1, 2], [3, 4]]> : tensor<2x2xi64>}> : (tensor<2x2xi64>) -> ()
      "check.expect_eq_const"(%8#2) <{value = dense<[[5, 6], [7, 8]]> : tensor<2x2xi64>}> : (tensor<2x2xi64>) -> ()
      "func.return"() : () -> ()
    }) : () -> ()
  }) : () -> ()
  "builtin.module"() <{sym_name = "cross_partition"}> ({
    "func.func"() <{function_type = (tensor<2x2xi64>) -> tensor<2x2xi64>, sym_name = "collective_permute"}> ({
    ^bb0(%arg0: tensor<2x2xi64>):
      %4 = "stablehlo.collective_permute"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, source_target_pairs = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>}> : (tensor<2x2xi64>) -> tensor<2x2xi64>
      "func.return"(%4) : (tensor<2x2xi64>) -> ()
    }) : () -> ()
    "func.func"() <{function_type = () -> (), sym_name = "main"}> ({
      %0 = "stablehlo.constant"() <{value = dense<[[1, 2], [3, 4]]> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
      %1 = "stablehlo.constant"() <{value = dense<[[5, 6], [7, 8]]> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
      %2 = "stablehlo.constant"() <{value = dense<[[9, 10], [11, 12]]> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
      %3:3 = "interpreter.run_parallel"(%0, %1, %2) <{programs = [[@collective_permute, @collective_permute, @collective_permute]]}> : (tensor<2x2xi64>, tensor<2x2xi64>, tensor<2x2xi64>) -> (tensor<2x2xi64>, tensor<2x2xi64>, tensor<2x2xi64>)
      "check.expect_eq_const"(%3#0) <{value = dense<0> : tensor<2x2xi64>}> : (tensor<2x2xi64>) -> ()
      "check.expect_eq_const"(%3#1) <{value = dense<[[1, 2], [3, 4]]> : tensor<2x2xi64>}> : (tensor<2x2xi64>) -> ()
      "check.expect_eq_const"(%3#2) <{value = dense<[[5, 6], [7, 8]]> : tensor<2x2xi64>}> : (tensor<2x2xi64>) -> ()
      "func.return"() : () -> ()
    }) : () -> ()
  }) : () -> ()
}) : () -> ()

