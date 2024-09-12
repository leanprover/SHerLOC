"builtin.module"() <{sym_name = "cross_replica"}> ({
  "func.func"() <{function_type = (tensor<2x4xi64>) -> tensor<2x2xi64>, sym_name = "reduce_scatter"}> ({
  ^bb0(%arg0: tensor<2x4xi64>):
    %3 = "stablehlo.reduce_scatter"(%arg0) <{replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, scatter_dimension = 1 : i64}> ({
    ^bb0(%arg1: tensor<i64>, %arg2: tensor<i64>):
      %4 = "stablehlo.add"(%arg1, %arg2) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      "stablehlo.return"(%4) : (tensor<i64>) -> ()
    }) : (tensor<2x4xi64>) -> tensor<2x2xi64>
    "func.return"(%3) : (tensor<2x2xi64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "main"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[1, 2, 3, 4], [5, 6, 7, 8]]> : tensor<2x4xi64>}> : () -> tensor<2x4xi64>
    %1 = "stablehlo.constant"() <{value = dense<[[9, 10, 11, 12], [13, 14, 15, 16]]> : tensor<2x4xi64>}> : () -> tensor<2x4xi64>
    %2:2 = "interpreter.run_parallel"(%0, %1) <{programs = [[@reduce_scatter], [@reduce_scatter]]}> : (tensor<2x4xi64>, tensor<2x4xi64>) -> (tensor<2x2xi64>, tensor<2x2xi64>)
    "check.expect_eq_const"(%2#0) <{value = dense<[[10, 12], [18, 20]]> : tensor<2x2xi64>}> : (tensor<2x2xi64>) -> ()
    "check.expect_eq_const"(%2#1) <{value = dense<[[14, 16], [22, 24]]> : tensor<2x2xi64>}> : (tensor<2x2xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() <{sym_name = "cross_replica_and_partition"}> ({
  "func.func"() <{function_type = (tensor<2x4xi64>) -> tensor<2x2xi64>, sym_name = "reduce_scatter"}> ({
  ^bb0(%arg0: tensor<2x4xi64>):
    %3 = "stablehlo.reduce_scatter"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, scatter_dimension = 1 : i64}> ({
    ^bb0(%arg1: tensor<i64>, %arg2: tensor<i64>):
      %4 = "stablehlo.add"(%arg1, %arg2) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      "stablehlo.return"(%4) : (tensor<i64>) -> ()
    }) : (tensor<2x4xi64>) -> tensor<2x2xi64>
    "func.return"(%3) : (tensor<2x2xi64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "main"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[1, 2, 3, 4], [5, 6, 7, 8]]> : tensor<2x4xi64>}> : () -> tensor<2x4xi64>
    %1 = "stablehlo.constant"() <{value = dense<[[9, 10, 11, 12], [13, 14, 15, 16]]> : tensor<2x4xi64>}> : () -> tensor<2x4xi64>
    %2:2 = "interpreter.run_parallel"(%0, %1) <{programs = [[@reduce_scatter], [@reduce_scatter]]}> : (tensor<2x4xi64>, tensor<2x4xi64>) -> (tensor<2x2xi64>, tensor<2x2xi64>)
    "check.expect_eq_const"(%2#0) <{value = dense<[[10, 12], [18, 20]]> : tensor<2x2xi64>}> : (tensor<2x2xi64>) -> ()
    "check.expect_eq_const"(%2#1) <{value = dense<[[14, 16], [22, 24]]> : tensor<2x2xi64>}> : (tensor<2x2xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() <{sym_name = "flattened_ids"}> ({
  "func.func"() <{function_type = (tensor<2x4xi64>) -> tensor<2x2xi64>, sym_name = "reduce_scatter"}> ({
  ^bb0(%arg0: tensor<2x4xi64>):
    %3 = "stablehlo.reduce_scatter"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, scatter_dimension = 1 : i64, use_global_device_ids}> ({
    ^bb0(%arg1: tensor<i64>, %arg2: tensor<i64>):
      %4 = "stablehlo.add"(%arg1, %arg2) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      "stablehlo.return"(%4) : (tensor<i64>) -> ()
    }) : (tensor<2x4xi64>) -> tensor<2x2xi64>
    "func.return"(%3) : (tensor<2x2xi64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "main"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[1, 2, 3, 4], [5, 6, 7, 8]]> : tensor<2x4xi64>}> : () -> tensor<2x4xi64>
    %1 = "stablehlo.constant"() <{value = dense<[[9, 10, 11, 12], [13, 14, 15, 16]]> : tensor<2x4xi64>}> : () -> tensor<2x4xi64>
    %2:2 = "interpreter.run_parallel"(%0, %1) <{programs = [[@reduce_scatter], [@reduce_scatter]]}> : (tensor<2x4xi64>, tensor<2x4xi64>) -> (tensor<2x2xi64>, tensor<2x2xi64>)
    "check.expect_eq_const"(%2#0) <{value = dense<[[10, 12], [18, 20]]> : tensor<2x2xi64>}> : (tensor<2x2xi64>) -> ()
    "check.expect_eq_const"(%2#1) <{value = dense<[[14, 16], [22, 24]]> : tensor<2x2xi64>}> : (tensor<2x2xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

