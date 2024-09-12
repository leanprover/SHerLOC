"builtin.module"() <{sym_name = "cross_replica"}> ({
  "func.func"() <{function_type = (tensor<1x2xi64>) -> tensor<1x2xi64>, sym_name = "collective_broadcast"}> ({
  ^bb0(%arg0: tensor<1x2xi64>):
    %5 = "stablehlo.collective_broadcast"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>, replica_groups = dense<[[2, 1]]> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> tensor<1x2xi64>
    "func.return"(%5) : (tensor<1x2xi64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "main"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[1, 2]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
    %1 = "stablehlo.constant"() <{value = dense<[[3, 4]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
    %2 = "stablehlo.constant"() <{value = dense<[[5, 6]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
    %3 = "stablehlo.constant"() <{value = dense<[[7, 8]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
    %4:4 = "interpreter.run_parallel"(%0, %1, %2, %3) <{programs = [[@collective_broadcast], [@collective_broadcast], [@collective_broadcast], [@collective_broadcast]]}> : (tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>) -> (tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>)
    "check.expect_eq_const"(%4#0) <{value = dense<0> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> ()
    "check.expect_eq_const"(%4#1) <{value = dense<[[5, 6]]> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> ()
    "check.expect_eq_const"(%4#2) <{value = dense<[[5, 6]]> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> ()
    "check.expect_eq_const"(%4#3) <{value = dense<0> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() <{sym_name = "cross_replica_multiple_output"}> ({
  "func.func"() <{function_type = (tensor<1x2xi64>) -> tensor<1x2xi64>, sym_name = "collective_broadcast"}> ({
  ^bb0(%arg0: tensor<1x2xi64>):
    %5 = "stablehlo.collective_broadcast"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>, replica_groups = dense<[[2, 1, 0]]> : tensor<1x3xi64>}> : (tensor<1x2xi64>) -> tensor<1x2xi64>
    "func.return"(%5) : (tensor<1x2xi64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "main"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[1, 2]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
    %1 = "stablehlo.constant"() <{value = dense<[[3, 4]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
    %2 = "stablehlo.constant"() <{value = dense<[[5, 6]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
    %3 = "stablehlo.constant"() <{value = dense<[[7, 8]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
    %4:4 = "interpreter.run_parallel"(%0, %1, %2, %3) <{programs = [[@collective_broadcast], [@collective_broadcast], [@collective_broadcast], [@collective_broadcast]]}> : (tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>) -> (tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>)
    "check.expect_eq_const"(%4#0) <{value = dense<[[5, 6]]> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> ()
    "check.expect_eq_const"(%4#1) <{value = dense<[[5, 6]]> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> ()
    "check.expect_eq_const"(%4#2) <{value = dense<[[5, 6]]> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> ()
    "check.expect_eq_const"(%4#3) <{value = dense<0> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() <{sym_name = "cross_replica_single_replica"}> ({
  "func.func"() <{function_type = (tensor<1x2xi64>) -> tensor<1x2xi64>, sym_name = "collective_broadcast"}> ({
  ^bb0(%arg0: tensor<1x2xi64>):
    %5 = "stablehlo.collective_broadcast"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>, replica_groups = dense<0> : tensor<1x1xi64>}> : (tensor<1x2xi64>) -> tensor<1x2xi64>
    "func.return"(%5) : (tensor<1x2xi64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "main"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[1, 2]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
    %1 = "stablehlo.constant"() <{value = dense<[[3, 4]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
    %2 = "stablehlo.constant"() <{value = dense<[[5, 6]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
    %3 = "stablehlo.constant"() <{value = dense<[[7, 8]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
    %4:4 = "interpreter.run_parallel"(%0, %1, %2, %3) <{programs = [[@collective_broadcast, @collective_broadcast, @collective_broadcast, @collective_broadcast]]}> : (tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>) -> (tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>)
    "check.expect_eq_const"(%4#0) <{value = dense<[[1, 2]]> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> ()
    "check.expect_eq_const"(%4#1) <{value = dense<[[3, 4]]> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> ()
    "check.expect_eq_const"(%4#2) <{value = dense<[[5, 6]]> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> ()
    "check.expect_eq_const"(%4#3) <{value = dense<[[7, 8]]> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() <{sym_name = "cross_replica_multiple_partitions"}> ({
  "func.func"() <{function_type = (tensor<1x2xi64>) -> tensor<1x2xi64>, sym_name = "collective_broadcast"}> ({
  ^bb0(%arg0: tensor<1x2xi64>):
    %5 = "stablehlo.collective_broadcast"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>, replica_groups = dense<[[1, 0]]> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> tensor<1x2xi64>
    "func.return"(%5) : (tensor<1x2xi64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "main"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[1, 2]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
    %1 = "stablehlo.constant"() <{value = dense<[[3, 4]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
    %2 = "stablehlo.constant"() <{value = dense<[[5, 6]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
    %3 = "stablehlo.constant"() <{value = dense<[[7, 8]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
    %4:4 = "interpreter.run_parallel"(%0, %1, %2, %3) <{programs = [[@collective_broadcast, @collective_broadcast], [@collective_broadcast, @collective_broadcast]]}> : (tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>) -> (tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>)
    "check.expect_eq_const"(%4#0) <{value = dense<[[5, 6]]> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> ()
    "check.expect_eq_const"(%4#1) <{value = dense<[[7, 8]]> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> ()
    "check.expect_eq_const"(%4#2) <{value = dense<[[5, 6]]> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> ()
    "check.expect_eq_const"(%4#3) <{value = dense<[[7, 8]]> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() <{sym_name = "cross_partition"}> ({
  "func.func"() <{function_type = (tensor<1x2xi64>) -> tensor<1x2xi64>, sym_name = "collective_broadcast"}> ({
  ^bb0(%arg0: tensor<1x2xi64>):
    %5 = "stablehlo.collective_broadcast"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, replica_groups = dense<[[2, 1]]> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> tensor<1x2xi64>
    "func.return"(%5) : (tensor<1x2xi64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "main"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[1, 2]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
    %1 = "stablehlo.constant"() <{value = dense<[[3, 4]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
    %2 = "stablehlo.constant"() <{value = dense<[[5, 6]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
    %3 = "stablehlo.constant"() <{value = dense<[[7, 8]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
    %4:4 = "interpreter.run_parallel"(%0, %1, %2, %3) <{programs = [[@collective_broadcast, @collective_broadcast, @collective_broadcast, @collective_broadcast]]}> : (tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>) -> (tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>)
    "check.expect_eq_const"(%4#0) <{value = dense<0> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> ()
    "check.expect_eq_const"(%4#1) <{value = dense<[[5, 6]]> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> ()
    "check.expect_eq_const"(%4#2) <{value = dense<[[5, 6]]> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> ()
    "check.expect_eq_const"(%4#3) <{value = dense<0> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() <{sym_name = "cross_partition_multiple_output"}> ({
  "func.func"() <{function_type = (tensor<1x2xi64>) -> tensor<1x2xi64>, sym_name = "collective_broadcast"}> ({
  ^bb0(%arg0: tensor<1x2xi64>):
    %5 = "stablehlo.collective_broadcast"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, replica_groups = dense<[[2, 1, 0]]> : tensor<1x3xi64>}> : (tensor<1x2xi64>) -> tensor<1x2xi64>
    "func.return"(%5) : (tensor<1x2xi64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "main"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[1, 2]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
    %1 = "stablehlo.constant"() <{value = dense<[[3, 4]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
    %2 = "stablehlo.constant"() <{value = dense<[[5, 6]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
    %3 = "stablehlo.constant"() <{value = dense<[[7, 8]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
    %4:4 = "interpreter.run_parallel"(%0, %1, %2, %3) <{programs = [[@collective_broadcast, @collective_broadcast, @collective_broadcast, @collective_broadcast]]}> : (tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>) -> (tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>)
    "check.expect_eq_const"(%4#0) <{value = dense<[[5, 6]]> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> ()
    "check.expect_eq_const"(%4#1) <{value = dense<[[5, 6]]> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> ()
    "check.expect_eq_const"(%4#2) <{value = dense<[[5, 6]]> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> ()
    "check.expect_eq_const"(%4#3) <{value = dense<0> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() <{sym_name = "cross_partition_single_partition"}> ({
  "func.func"() <{function_type = (tensor<1x2xi64>) -> tensor<1x2xi64>, sym_name = "collective_broadcast"}> ({
  ^bb0(%arg0: tensor<1x2xi64>):
    %5 = "stablehlo.collective_broadcast"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, replica_groups = dense<0> : tensor<1x1xi64>}> : (tensor<1x2xi64>) -> tensor<1x2xi64>
    "func.return"(%5) : (tensor<1x2xi64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "main"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[1, 2]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
    %1 = "stablehlo.constant"() <{value = dense<[[3, 4]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
    %2 = "stablehlo.constant"() <{value = dense<[[5, 6]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
    %3 = "stablehlo.constant"() <{value = dense<[[7, 8]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
    %4:4 = "interpreter.run_parallel"(%0, %1, %2, %3) <{programs = [[@collective_broadcast], [@collective_broadcast], [@collective_broadcast], [@collective_broadcast]]}> : (tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>) -> (tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>)
    "check.expect_eq_const"(%4#0) <{value = dense<[[1, 2]]> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> ()
    "check.expect_eq_const"(%4#1) <{value = dense<[[3, 4]]> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> ()
    "check.expect_eq_const"(%4#2) <{value = dense<[[5, 6]]> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> ()
    "check.expect_eq_const"(%4#3) <{value = dense<[[7, 8]]> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() <{sym_name = "cross_partition_multiple_replicas"}> ({
  "func.func"() <{function_type = (tensor<1x2xi64>) -> tensor<1x2xi64>, sym_name = "collective_broadcast"}> ({
  ^bb0(%arg0: tensor<1x2xi64>):
    %5 = "stablehlo.collective_broadcast"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, replica_groups = dense<[[1, 0]]> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> tensor<1x2xi64>
    "func.return"(%5) : (tensor<1x2xi64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "main"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[1, 2]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
    %1 = "stablehlo.constant"() <{value = dense<[[3, 4]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
    %2 = "stablehlo.constant"() <{value = dense<[[5, 6]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
    %3 = "stablehlo.constant"() <{value = dense<[[7, 8]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
    %4:4 = "interpreter.run_parallel"(%0, %1, %2, %3) <{programs = [[@collective_broadcast, @collective_broadcast], [@collective_broadcast, @collective_broadcast]]}> : (tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>) -> (tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>)
    "check.expect_eq_const"(%4#0) <{value = dense<[[3, 4]]> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> ()
    "check.expect_eq_const"(%4#1) <{value = dense<[[3, 4]]> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> ()
    "check.expect_eq_const"(%4#2) <{value = dense<[[7, 8]]> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> ()
    "check.expect_eq_const"(%4#3) <{value = dense<[[7, 8]]> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

