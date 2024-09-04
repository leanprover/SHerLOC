"builtin.module"() ({
  "builtin.module"() <{sym_name = "cross_replica"}> ({
    "func.func"() <{function_type = (tensor<2x2xi64>) -> tensor<2x4xi64>, sym_name = "all_gather"}> ({
    ^bb0(%arg5: tensor<2x2xi64>):
      %21 = "stablehlo.all_gather"(%arg5) <{all_gather_dim = 1 : i64, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>}> : (tensor<2x2xi64>) -> tensor<2x4xi64>
      "func.return"(%21) : (tensor<2x4xi64>) -> ()
    }) : () -> ()
    "func.func"() <{function_type = () -> (), sym_name = "main"}> ({
      %18 = "stablehlo.constant"() <{value = dense<[[1, 2], [3, 4]]> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
      %19 = "stablehlo.constant"() <{value = dense<[[5, 6], [7, 8]]> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
      %20:2 = "interpreter.run_parallel"(%18, %19) <{programs = [[@all_gather], [@all_gather]]}> : (tensor<2x2xi64>, tensor<2x2xi64>) -> (tensor<2x4xi64>, tensor<2x4xi64>)
      "check.expect_eq_const"(%20#0) <{value = dense<[[1, 2, 5, 6], [3, 4, 7, 8]]> : tensor<2x4xi64>}> : (tensor<2x4xi64>) -> ()
      "check.expect_eq_const"(%20#1) <{value = dense<[[1, 2, 5, 6], [3, 4, 7, 8]]> : tensor<2x4xi64>}> : (tensor<2x4xi64>) -> ()
      "func.return"() : () -> ()
    }) : () -> ()
  }) : () -> ()
  "builtin.module"() <{sym_name = "cross_replica_and_partition"}> ({
    "func.func"() <{function_type = (tensor<2x2xi64>) -> tensor<2x4xi64>, sym_name = "all_gather"}> ({
    ^bb0(%arg4: tensor<2x2xi64>):
      %17 = "stablehlo.all_gather"(%arg4) <{all_gather_dim = 1 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>}> : (tensor<2x2xi64>) -> tensor<2x4xi64>
      "func.return"(%17) : (tensor<2x4xi64>) -> ()
    }) : () -> ()
    "func.func"() <{function_type = () -> (), sym_name = "main"}> ({
      %14 = "stablehlo.constant"() <{value = dense<[[1, 2], [3, 4]]> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
      %15 = "stablehlo.constant"() <{value = dense<[[5, 6], [7, 8]]> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
      %16:2 = "interpreter.run_parallel"(%14, %15) <{programs = [[@all_gather], [@all_gather]]}> : (tensor<2x2xi64>, tensor<2x2xi64>) -> (tensor<2x4xi64>, tensor<2x4xi64>)
      "check.expect_eq_const"(%16#0) <{value = dense<[[1, 2, 5, 6], [3, 4, 7, 8]]> : tensor<2x4xi64>}> : (tensor<2x4xi64>) -> ()
      "check.expect_eq_const"(%16#1) <{value = dense<[[1, 2, 5, 6], [3, 4, 7, 8]]> : tensor<2x4xi64>}> : (tensor<2x4xi64>) -> ()
      "func.return"() : () -> ()
    }) : () -> ()
  }) : () -> ()
  "builtin.module"() <{sym_name = "cross_replica_and_partition_issue_1933"}> ({
    "func.func"() <{function_type = (tensor<2x2xi64>) -> tensor<2x8xi64>, sym_name = "all_gather"}> ({
    ^bb0(%arg3: tensor<2x2xi64>):
      %13 = "stablehlo.all_gather"(%arg3) <{all_gather_dim = 1 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>}> : (tensor<2x2xi64>) -> tensor<2x8xi64>
      "func.return"(%13) : (tensor<2x8xi64>) -> ()
    }) : () -> ()
    "func.func"() <{function_type = () -> (), sym_name = "main"}> ({
      %10 = "stablehlo.constant"() <{value = dense<[[1, 2], [3, 4]]> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
      %11 = "stablehlo.constant"() <{value = dense<[[5, 6], [7, 8]]> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
      %12:4 = "interpreter.run_parallel"(%11, %11, %10, %11) <{programs = [[@all_gather, @all_gather], [@all_gather, @all_gather]]}> : (tensor<2x2xi64>, tensor<2x2xi64>, tensor<2x2xi64>, tensor<2x2xi64>) -> (tensor<2x8xi64>, tensor<2x8xi64>, tensor<2x8xi64>, tensor<2x8xi64>)
      "check.expect_eq_const"(%12#0) <{value = dense<[[5, 6, 1, 2, 5, 6, 5, 6], [7, 8, 3, 4, 7, 8, 7, 8]]> : tensor<2x8xi64>}> : (tensor<2x8xi64>) -> ()
      "check.expect_eq_const"(%12#1) <{value = dense<[[5, 6, 1, 2, 5, 6, 5, 6], [7, 8, 3, 4, 7, 8, 7, 8]]> : tensor<2x8xi64>}> : (tensor<2x8xi64>) -> ()
      "check.expect_eq_const"(%12#2) <{value = dense<[[5, 6, 1, 2, 5, 6, 5, 6], [7, 8, 3, 4, 7, 8, 7, 8]]> : tensor<2x8xi64>}> : (tensor<2x8xi64>) -> ()
      "check.expect_eq_const"(%12#3) <{value = dense<[[5, 6, 1, 2, 5, 6, 5, 6], [7, 8, 3, 4, 7, 8, 7, 8]]> : tensor<2x8xi64>}> : (tensor<2x8xi64>) -> ()
      "func.return"() : () -> ()
    }) : () -> ()
  }) : () -> ()
  "builtin.module"() <{sym_name = "flattened_ids"}> ({
    "func.func"() <{function_type = (tensor<2x2xi64>) -> tensor<2x4xi64>, sym_name = "all_gather"}> ({
    ^bb0(%arg2: tensor<2x2xi64>):
      %9 = "stablehlo.all_gather"(%arg2) <{all_gather_dim = 1 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, use_global_device_ids}> : (tensor<2x2xi64>) -> tensor<2x4xi64>
      "func.return"(%9) : (tensor<2x4xi64>) -> ()
    }) : () -> ()
    "func.func"() <{function_type = () -> (), sym_name = "main"}> ({
      %6 = "stablehlo.constant"() <{value = dense<[[1, 2], [3, 4]]> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
      %7 = "stablehlo.constant"() <{value = dense<[[5, 6], [7, 8]]> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
      %8:2 = "interpreter.run_parallel"(%6, %7) <{programs = [[@all_gather], [@all_gather]]}> : (tensor<2x2xi64>, tensor<2x2xi64>) -> (tensor<2x4xi64>, tensor<2x4xi64>)
      "check.expect_eq_const"(%8#0) <{value = dense<[[1, 2, 5, 6], [3, 4, 7, 8]]> : tensor<2x4xi64>}> : (tensor<2x4xi64>) -> ()
      "check.expect_eq_const"(%8#1) <{value = dense<[[1, 2, 5, 6], [3, 4, 7, 8]]> : tensor<2x4xi64>}> : (tensor<2x4xi64>) -> ()
      "func.return"() : () -> ()
    }) : () -> ()
  }) : () -> ()
  "builtin.module"() <{sym_name = "cross_replica_variadic_inputs"}> ({
    "func.func"() <{function_type = (tensor<2x2xi64>, tensor<2x2xi32>) -> (tensor<2x4xi64>, tensor<2x4xi32>), sym_name = "all_gather"}> ({
    ^bb0(%arg0: tensor<2x2xi64>, %arg1: tensor<2x2xi32>):
      %5:2 = "stablehlo.all_gather"(%arg0, %arg1) <{all_gather_dim = 1 : i64, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>}> : (tensor<2x2xi64>, tensor<2x2xi32>) -> (tensor<2x4xi64>, tensor<2x4xi32>)
      "func.return"(%5#0, %5#1) : (tensor<2x4xi64>, tensor<2x4xi32>) -> ()
    }) : () -> ()
    "func.func"() <{function_type = () -> (), sym_name = "main"}> ({
      %0 = "stablehlo.constant"() <{value = dense<[[1, 2], [3, 4]]> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
      %1 = "stablehlo.constant"() <{value = dense<[[5, 6], [7, 8]]> : tensor<2x2xi32>}> : () -> tensor<2x2xi32>
      %2 = "stablehlo.constant"() <{value = dense<[[11, 12], [13, 14]]> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
      %3 = "stablehlo.constant"() <{value = dense<[[15, 16], [17, 18]]> : tensor<2x2xi32>}> : () -> tensor<2x2xi32>
      %4:4 = "interpreter.run_parallel"(%0, %1, %2, %3) <{programs = [[@all_gather], [@all_gather]]}> : (tensor<2x2xi64>, tensor<2x2xi32>, tensor<2x2xi64>, tensor<2x2xi32>) -> (tensor<2x4xi64>, tensor<2x4xi32>, tensor<2x4xi64>, tensor<2x4xi32>)
      "check.expect_eq_const"(%4#0) <{value = dense<[[1, 2, 11, 12], [3, 4, 13, 14]]> : tensor<2x4xi64>}> : (tensor<2x4xi64>) -> ()
      "check.expect_eq_const"(%4#1) <{value = dense<[[5, 6, 15, 16], [7, 8, 17, 18]]> : tensor<2x4xi32>}> : (tensor<2x4xi32>) -> ()
      "check.expect_eq_const"(%4#2) <{value = dense<[[1, 2, 11, 12], [3, 4, 13, 14]]> : tensor<2x4xi64>}> : (tensor<2x4xi64>) -> ()
      "check.expect_eq_const"(%4#3) <{value = dense<[[5, 6, 15, 16], [7, 8, 17, 18]]> : tensor<2x4xi32>}> : (tensor<2x4xi32>) -> ()
      "func.return"() : () -> ()
    }) : () -> ()
  }) : () -> ()
}) : () -> ()

