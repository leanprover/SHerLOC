"builtin.module"() <{sym_name = "cross_replica"}> ({
  "func.func"() <{function_type = (tensor<2x2xi64>) -> tensor<2x4xi64>, sym_name = "all_gather"}> ({
  ^bb0(%arg0: tensor<2x2xi64>):
    %3 = "stablehlo.all_gather"(%arg0) <{all_gather_dim = 1 : i64, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>}> : (tensor<2x2xi64>) -> tensor<2x4xi64>
    "func.return"(%3) : (tensor<2x4xi64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "main"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[1, 2], [3, 4]]> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
    %1 = "stablehlo.constant"() <{value = dense<[[5, 6], [7, 8]]> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
    %2:2 = "interpreter.run_parallel"(%0, %1) <{programs = [[@all_gather], [@all_gather]]}> : (tensor<2x2xi64>, tensor<2x2xi64>) -> (tensor<2x4xi64>, tensor<2x4xi64>)
    "check.expect_eq_const"(%2#0) <{value = dense<[[1, 2, 5, 6], [3, 4, 7, 8]]> : tensor<2x4xi64>}> : (tensor<2x4xi64>) -> ()
    "check.expect_eq_const"(%2#1) <{value = dense<[[1, 2, 5, 6], [3, 4, 7, 8]]> : tensor<2x4xi64>}> : (tensor<2x4xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() <{sym_name = "cross_replica_and_partition"}> ({
  "func.func"() <{function_type = (tensor<2x2xi64>) -> tensor<2x4xi64>, sym_name = "all_gather"}> ({
  ^bb0(%arg0: tensor<2x2xi64>):
    %3 = "stablehlo.all_gather"(%arg0) <{all_gather_dim = 1 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>}> : (tensor<2x2xi64>) -> tensor<2x4xi64>
    "func.return"(%3) : (tensor<2x4xi64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "main"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[1, 2], [3, 4]]> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
    %1 = "stablehlo.constant"() <{value = dense<[[5, 6], [7, 8]]> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
    %2:2 = "interpreter.run_parallel"(%0, %1) <{programs = [[@all_gather], [@all_gather]]}> : (tensor<2x2xi64>, tensor<2x2xi64>) -> (tensor<2x4xi64>, tensor<2x4xi64>)
    "check.expect_eq_const"(%2#0) <{value = dense<[[1, 2, 5, 6], [3, 4, 7, 8]]> : tensor<2x4xi64>}> : (tensor<2x4xi64>) -> ()
    "check.expect_eq_const"(%2#1) <{value = dense<[[1, 2, 5, 6], [3, 4, 7, 8]]> : tensor<2x4xi64>}> : (tensor<2x4xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() <{sym_name = "cross_replica_and_partition_issue_1933"}> ({
  "func.func"() <{function_type = (tensor<2x2xi64>) -> tensor<2x8xi64>, sym_name = "all_gather"}> ({
  ^bb0(%arg0: tensor<2x2xi64>):
    %3 = "stablehlo.all_gather"(%arg0) <{all_gather_dim = 1 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>}> : (tensor<2x2xi64>) -> tensor<2x8xi64>
    "func.return"(%3) : (tensor<2x8xi64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "main"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[1, 2], [3, 4]]> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
    %1 = "stablehlo.constant"() <{value = dense<[[5, 6], [7, 8]]> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
    %2:4 = "interpreter.run_parallel"(%1, %1, %0, %1) <{programs = [[@all_gather, @all_gather], [@all_gather, @all_gather]]}> : (tensor<2x2xi64>, tensor<2x2xi64>, tensor<2x2xi64>, tensor<2x2xi64>) -> (tensor<2x8xi64>, tensor<2x8xi64>, tensor<2x8xi64>, tensor<2x8xi64>)
    "check.expect_eq_const"(%2#0) <{value = dense<[[5, 6, 1, 2, 5, 6, 5, 6], [7, 8, 3, 4, 7, 8, 7, 8]]> : tensor<2x8xi64>}> : (tensor<2x8xi64>) -> ()
    "check.expect_eq_const"(%2#1) <{value = dense<[[5, 6, 1, 2, 5, 6, 5, 6], [7, 8, 3, 4, 7, 8, 7, 8]]> : tensor<2x8xi64>}> : (tensor<2x8xi64>) -> ()
    "check.expect_eq_const"(%2#2) <{value = dense<[[5, 6, 1, 2, 5, 6, 5, 6], [7, 8, 3, 4, 7, 8, 7, 8]]> : tensor<2x8xi64>}> : (tensor<2x8xi64>) -> ()
    "check.expect_eq_const"(%2#3) <{value = dense<[[5, 6, 1, 2, 5, 6, 5, 6], [7, 8, 3, 4, 7, 8, 7, 8]]> : tensor<2x8xi64>}> : (tensor<2x8xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() <{sym_name = "flattened_ids"}> ({
  "func.func"() <{function_type = (tensor<2x2xi64>) -> tensor<2x4xi64>, sym_name = "all_gather"}> ({
  ^bb0(%arg0: tensor<2x2xi64>):
    %3 = "stablehlo.all_gather"(%arg0) <{all_gather_dim = 1 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, use_global_device_ids}> : (tensor<2x2xi64>) -> tensor<2x4xi64>
    "func.return"(%3) : (tensor<2x4xi64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "main"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[1, 2], [3, 4]]> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
    %1 = "stablehlo.constant"() <{value = dense<[[5, 6], [7, 8]]> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
    %2:2 = "interpreter.run_parallel"(%0, %1) <{programs = [[@all_gather], [@all_gather]]}> : (tensor<2x2xi64>, tensor<2x2xi64>) -> (tensor<2x4xi64>, tensor<2x4xi64>)
    "check.expect_eq_const"(%2#0) <{value = dense<[[1, 2, 5, 6], [3, 4, 7, 8]]> : tensor<2x4xi64>}> : (tensor<2x4xi64>) -> ()
    "check.expect_eq_const"(%2#1) <{value = dense<[[1, 2, 5, 6], [3, 4, 7, 8]]> : tensor<2x4xi64>}> : (tensor<2x4xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
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

